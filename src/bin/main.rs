use lancy::codegen::{
    isa::x64::{inst::X64Inst, regs::*},
    tir::{BlockData, Func, RegClass},
};

fn main() {
    let mut func = Func::<X64Inst>::new("foo".to_string());
    let b0 = func.add_empty_block();
    let v0 = func.new_vreg(RegClass::Int(8));
    let v1 = func.new_vreg(RegClass::Int(8));
    let b1 = func.add_empty_block();
    let b2 = func.add_empty_block();

    {
        let block_data = func.get_block_data_mut(b0);
        block_data.push(X64Inst::Mov64rr { dst: v0, src: RAX });
        block_data.push(X64Inst::Jmp { dst: b1 });
    }

    {
        let block_data = func.get_block_data_mut(b1);
        block_data.push(X64Inst::Mov64rr { dst: v1, src: v0 });
        block_data.push(X64Inst::Jmp { dst: b2 });
    }

    {
        let block_data = func.get_block_data_mut(b2);
        block_data.push(X64Inst::Mov64rr { dst: RAX, src: v1 });
        block_data.push(X64Inst::Jmp { dst: b0 });
    }

    func.construct_cfg().unwrap();

    println!("{func}");
}
