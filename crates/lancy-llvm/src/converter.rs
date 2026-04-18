//! LLVM IR -> lancy IR conversion.

use std::collections::HashMap;

use inkwell::IntPredicate;
use inkwell::basic_block::BasicBlock;
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::values::{
    AsValueRef, BasicValueEnum, FunctionValue, InstructionOpcode, InstructionValue,
};

use lancy::codegen::isa::x64::builder::FuncBuilder;
use lancy::codegen::isa::x64::inst::{Cond, X64Inst};
use lancy::codegen::isa::x64::pipeline;
use lancy::codegen::jit::Module as JitModule;
use lancy::codegen::tir::{Block, Func, Reg};

use crate::error::ConvertError;

/// Pointer-identity key for an LLVM value / basic block handle.
type PtrKey = usize;

fn val_key<V: AsValueRef>(v: &V) -> PtrKey {
    v.as_value_ref() as usize
}

fn bb_key(bb: BasicBlock) -> PtrKey {
    // BasicBlock is a newtype over LLVMBasicBlockRef; `as_mut_ptr` exposes
    // that raw pointer, which is stable for the lifetime of the Context.
    bb.as_mut_ptr() as usize
}

/// Parse `ir` (textual LLVM IR or LLVM bitcode) and lower the function
/// named `name` into a lancy `Func<X64Inst>`.
pub fn convert_function(ir: &[u8], name: &str) -> Result<Func<X64Inst>, ConvertError> {
    let ctx = Context::create();
    // inkwell's memory-buffer APIs require a trailing nul byte; callers
    // typically pass slices from `Vec<u8>` or string literals without
    // one, so we paper over the difference here.
    let owned: Vec<u8> = if ir.last() == Some(&0) {
        ir.to_vec()
    } else {
        let mut v = Vec::with_capacity(ir.len() + 1);
        v.extend_from_slice(ir);
        v.push(0);
        v
    };
    let mem_buf = MemoryBuffer::create_from_memory_range_copy(&owned, "lancy_llvm_input");
    let module = ctx
        .create_module_from_ir(mem_buf)
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    let func = module
        .get_function(name)
        .ok_or_else(|| ConvertError::FunctionNotFound(name.to_string()))?;
    let mut cv = Converter::new(name);
    cv.lower(func)?;
    Ok(cv.finish())
}

/// Parse `ir`, lower `name`, run the x64 pipeline, and load into a JIT
/// module. The caller owns the `JitModule` and can call `entry()` to get
/// a `sysv64` function pointer.
pub fn compile_ir_to_jit(ir: &[u8], name: &str) -> Result<JitModule, ConvertError> {
    let func = convert_function(ir, name)?;
    let m = pipeline::jit(func)?;
    Ok(m)
}

struct Converter {
    builder: FuncBuilder,
    /// Map from LLVM value raw pointer to the lancy vreg holding its value.
    vals: HashMap<PtrKey, Reg>,
    /// Map from LLVM basic block raw pointer to the lancy `Block`.
    blocks: HashMap<PtrKey, Block>,
}

impl Converter {
    fn new(name: &str) -> Self {
        Self {
            builder: FuncBuilder::new(name),
            vals: HashMap::new(),
            blocks: HashMap::new(),
        }
    }

    fn finish(self) -> Func<X64Inst> {
        self.builder.build()
    }

    fn lower(&mut self, func: FunctionValue<'_>) -> Result<(), ConvertError> {
        let bbs = func.get_basic_blocks();
        if bbs.is_empty() {
            return Err(ConvertError::Malformed(
                "function has no basic blocks (declaration only?)".into(),
            ));
        }

        // Map LLVM blocks to lancy blocks. The first LLVM block is the
        // entry and maps to `FuncBuilder`'s pre-created entry block.
        self.blocks.insert(bb_key(bbs[0]), self.builder.entry_block());
        for bb in &bbs[1..] {
            let nb = self.builder.new_block();
            self.blocks.insert(bb_key(*bb), nb);
        }

        // Emit `arg` pseudos in entry-block order for each function param.
        self.builder.switch_to_block(self.builder.entry_block());
        for param in func.get_param_iter() {
            check_int64(param)?;
            let r = self.builder.arg();
            self.vals.insert(val_key(&param), r);
        }

        for bb in &bbs {
            let lancy_bb = self.blocks[&bb_key(*bb)];
            self.builder.switch_to_block(lancy_bb);
            self.lower_block(*bb)?;
        }
        Ok(())
    }

    fn lower_block(&mut self, bb: BasicBlock<'_>) -> Result<(), ConvertError> {
        let insts: Vec<InstructionValue> = bb.get_instructions().collect();
        if insts.is_empty() {
            return Err(ConvertError::Malformed(
                "basic block has no terminator".into(),
            ));
        }

        // Detect icmp+br fusion: if the terminator is a conditional br
        // whose condition is the immediately-preceding icmp, translate
        // the pair as a single `branch_icmp` and skip the icmp's own
        // lowering. This is the pattern clang emits at every opt level
        // for `if (a OP b)`-style control flow.
        let fuse_idx = fused_icmp_index(&insts);

        for (i, inst) in insts.iter().enumerate() {
            if Some(i) == fuse_idx {
                continue;
            }
            if let Some(icmp_idx) = fuse_idx
                && inst.get_opcode() == InstructionOpcode::Br
            {
                self.lower_fused_icmp_br(insts[icmp_idx], *inst)?;
                continue;
            }
            self.lower_inst(*inst)?;
        }
        Ok(())
    }

    fn lower_inst(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        use InstructionOpcode as Op;
        match inst.get_opcode() {
            Op::Add => self.lower_binop(inst, BinOp::Add),
            Op::Sub => self.lower_binop(inst, BinOp::Sub),
            Op::Mul => self.lower_binop(inst, BinOp::Mul),
            Op::Return => self.lower_ret(inst),
            Op::Br => self.lower_br(inst),
            Op::SExt | Op::ZExt | Op::Trunc | Op::BitCast => self.lower_passthrough(inst),
            other => Err(ConvertError::Unsupported(format!(
                "instruction opcode {other:?}"
            ))),
        }
    }

    fn lower_binop(
        &mut self,
        inst: InstructionValue<'_>,
        op: BinOp,
    ) -> Result<(), ConvertError> {
        let lhs = self.operand_reg(inst, 0)?;
        let rhs = self.operand_reg(inst, 1)?;
        let dst = match op {
            BinOp::Add => self.builder.add(lhs, rhs),
            BinOp::Sub => self.builder.sub(lhs, rhs),
            BinOp::Mul => self.builder.imul(lhs, rhs),
        };
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    fn lower_ret(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        if inst.get_num_operands() == 0 {
            return Err(ConvertError::Unsupported(
                "void return — lancy requires an i64 return value".into(),
            ));
        }
        let r = self.operand_reg(inst, 0)?;
        self.builder.ret(r);
        Ok(())
    }

    fn lower_br(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        match inst.get_num_operands() {
            1 => {
                // `br label %dst`
                let target = inst
                    .get_operand(0)
                    .and_then(inkwell::values::Operand::block)
                    .ok_or_else(|| {
                        ConvertError::Malformed("br missing label operand".into())
                    })?;
                let blk = self.blocks[&bb_key(target)];
                self.builder.jmp(blk);
                Ok(())
            }
            3 => Err(ConvertError::Unsupported(
                "conditional br not preceded by icmp (likely a non-fused boolean path)"
                    .into(),
            )),
            n => Err(ConvertError::Malformed(format!(
                "br with {n} operands"
            ))),
        }
    }

    fn lower_passthrough(
        &mut self,
        inst: InstructionValue<'_>,
    ) -> Result<(), ConvertError> {
        // lancy only carries i64 right now — narrower-to-wider and
        // wider-to-narrower int casts and bitcasts on integers become
        // identity copies at the value level. The emitted bytes of
        // upstream ops already leave the register in the low-64 form
        // lancy uses.
        let src = self.operand_reg(inst, 0)?;
        self.vals.insert(val_key(&inst), src);
        Ok(())
    }

    fn lower_fused_icmp_br(
        &mut self,
        icmp: InstructionValue<'_>,
        br: InstructionValue<'_>,
    ) -> Result<(), ConvertError> {
        let pred = icmp
            .get_icmp_predicate()
            .ok_or_else(|| ConvertError::Malformed("icmp without predicate".into()))?;
        let cond = cond_from_predicate(pred);
        let a = self.operand_reg(icmp, 0)?;
        let b = self.operand_reg(icmp, 1)?;
        // LLVM C API conditional br operand order is (cond, false, true).
        let false_bb = br
            .get_operand(1)
            .and_then(inkwell::values::Operand::block)
            .ok_or_else(|| ConvertError::Malformed("cond br missing false label".into()))?;
        let true_bb = br
            .get_operand(2)
            .and_then(inkwell::values::Operand::block)
            .ok_or_else(|| ConvertError::Malformed("cond br missing true label".into()))?;
        let taken = self.blocks[&bb_key(true_bb)];
        let not_taken = self.blocks[&bb_key(false_bb)];
        self.builder.branch_icmp(cond, a, b, taken, not_taken);
        Ok(())
    }

    /// Resolve operand `idx` of `inst` to a lancy vreg. If the operand is
    /// an as-yet-unmaterialized integer constant, materialize it with
    /// `iconst64` now and cache the result.
    fn operand_reg(
        &mut self,
        inst: InstructionValue<'_>,
        idx: u32,
    ) -> Result<Reg, ConvertError> {
        let operand = inst
            .get_operand(idx)
            .and_then(inkwell::values::Operand::value)
            .ok_or_else(|| {
                ConvertError::Malformed(format!(
                    "instruction {:?} missing value operand {idx}",
                    inst.get_opcode()
                ))
            })?;
        self.resolve_value(operand)
    }

    fn resolve_value(&mut self, v: BasicValueEnum<'_>) -> Result<Reg, ConvertError> {
        let key = val_key(&v);
        if let Some(&r) = self.vals.get(&key) {
            return Ok(r);
        }
        // Only integer constants reach this path — any other unmapped
        // operand indicates a construct the converter doesn't handle.
        let BasicValueEnum::IntValue(iv) = v else {
            return Err(ConvertError::Unsupported(format!(
                "non-integer value operand: {v:?}"
            )));
        };
        if !iv.is_const() {
            return Err(ConvertError::Unsupported(format!(
                "non-constant unmapped integer value: {v:?}"
            )));
        }
        let imm = iv.get_sign_extended_constant().ok_or_else(|| {
            ConvertError::Unsupported(format!("int constant wider than 64 bits: {v:?}"))
        })?;
        let r = self.builder.iconst64(imm);
        self.vals.insert(key, r);
        Ok(r)
    }
}

#[derive(Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
}

fn cond_from_predicate(p: IntPredicate) -> Cond {
    match p {
        IntPredicate::EQ => Cond::Z,
        IntPredicate::NE => Cond::NZ,
        IntPredicate::SLT => Cond::L,
        IntPredicate::SLE => Cond::LE,
        IntPredicate::SGT => Cond::G,
        IntPredicate::SGE => Cond::GE,
        IntPredicate::ULT => Cond::B,
        IntPredicate::ULE => Cond::BE,
        IntPredicate::UGT => Cond::A,
        IntPredicate::UGE => Cond::AE,
    }
}

/// Return the index of an icmp that should be fused with the block's
/// conditional-br terminator. We fuse only when the icmp immediately
/// precedes the br and the br consumes its result — the common pattern
/// clang emits. Any other layout (intervening instructions, icmp with
/// multiple uses, icmp in a different block) is translated as-is, which
/// currently fails because the builder has no way to materialize an i1.
fn fused_icmp_index(insts: &[InstructionValue<'_>]) -> Option<usize> {
    if insts.len() < 2 {
        return None;
    }
    let n = insts.len();
    let br = insts[n - 1];
    if br.get_opcode() != InstructionOpcode::Br || br.get_num_operands() != 3 {
        return None;
    }
    let cond_val = br.get_operand(0).and_then(inkwell::values::Operand::value)?;
    let cond_key = val_key(&cond_val);
    let prev = insts[n - 2];
    if prev.get_opcode() != InstructionOpcode::ICmp {
        return None;
    }
    if val_key(&prev) != cond_key {
        return None;
    }
    Some(n - 2)
}

fn check_int64(v: BasicValueEnum<'_>) -> Result<(), ConvertError> {
    match v {
        BasicValueEnum::IntValue(iv) => {
            let bw = iv.get_type().get_bit_width();
            if bw == 64 {
                Ok(())
            } else {
                Err(ConvertError::Unsupported(format!(
                    "function argument is i{bw}; only i64 is supported"
                )))
            }
        }
        other => Err(ConvertError::Unsupported(format!(
            "non-integer function argument: {other:?}"
        ))),
    }
}
