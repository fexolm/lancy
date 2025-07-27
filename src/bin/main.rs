use lancy::codegen::{
    isa::x64::{backend::X64Backend, inst::X64Inst},
    tir::{BlockData, Func, RegClass},
};

fn main() {
    let mut func = Func::<X64Backend>::new(
        "foo".to_string(),
        vec![RegClass::Int(8)],
        vec![RegClass::Int(8)],
    );

    let block1 = func.add_empty_block();

    let block2 = {
        let mut block_data = BlockData::new();

        let src = func.get_arg(0);
        let dst = func.get_result(0);

        block_data.push(X64Inst::Mov64rr { src, dst });
        block_data.push(X64Inst::Ret);

        func.add_block(block_data)
    };

    {
        let b1_data = func.get_block_data_mut(block1);
        b1_data.push(X64Inst::Jmp { dst: block2 });
    }

    let _ = func.get_dfg();

    println!("{func}");
}
