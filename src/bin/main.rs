use lancy::codegen::{isa::x64::inst::X64Inst, tir::{BlockData, Func, RegClass}};

fn main() {
    let mut func = Func::new();
    let _block = {
        let mut block_data = BlockData::new();

        let src = func.new_vreg(RegClass::Int(8));
        let dst = func.new_vreg(RegClass::Int(8));

        block_data.push(X64Inst::Mov64rr { src, dst });
        block_data.push(X64Inst::Ret);

        func.add_block(block_data)
    };
}
