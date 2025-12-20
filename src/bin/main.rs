use lancy::codegen::regalloc::{RegAlloc, RegisterBinding};
use lancy::codegen::{
    analysis::cfg::CFG,
    isa::x64::{inst::X64Inst, regs::*},
    tir::Func,
};

fn main() {
    let mut func = Func::<X64Inst>::new("foo".to_string());
    let b0 = func.add_empty_block();
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();
    let b1 = func.add_empty_block();
    let b2 = func.add_empty_block();

    {
        let block_data = func.get_block_data_mut(b0);
        block_data.push_target_inst(X64Inst::Mov64rr { dst: v0, src: v2 });
        block_data.push_target_inst(X64Inst::Jmp { dst: b1 });
    }

    {
        let block_data = func.get_block_data_mut(b1);
        block_data.push_target_inst(X64Inst::Mov64rr { dst: v1, src: v0 });
        block_data.push_target_inst(X64Inst::Jmp { dst: b2 });
    }

    {
        let block_data = func.get_block_data_mut(b2);
        block_data.push_target_inst(X64Inst::Mov64rr { dst: v3, src: v1 });
        block_data.push_target_inst(X64Inst::Ret { src: v3 });
    }

    let cfg = CFG::compute(&func).unwrap();

    let mut reg_bind = RegisterBinding::new(func.get_regs_count());
    reg_bind.add(v2, RAX);
    reg_bind.add(v3, RAX);

    println!("{func}");

    // let mut regalloc = RegAlloc::new(&func, &cfg);
    // let regalloc_intervals = regalloc.run();
    // apply_regalloc_result(&mut func, regalloc_intervals);

    println!("{func}");
}
