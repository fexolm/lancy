use lancy::codegen::isa::x64::builder::FuncBuilder;
use lancy::codegen::isa::x64::pipeline;

fn main() {
    let mut b = FuncBuilder::new("add");
    let x = b.arg();
    let y = b.arg();
    let s = b.add(x, y);
    b.ret(s);
    let func = b.build();
    println!("TIR:\n{func}");

    let module = pipeline::jit(func).expect("JIT compile+load");
    println!("code @ {:p}, {} bytes mapped", module.code_ptr(), module.size());

    type Add = unsafe extern "sysv64" fn(i64, i64) -> i64;
    // SAFETY: the compiled function has exactly this signature.
    let f: Add = unsafe { module.entry() };
    println!("add(2, 3) = {}", unsafe { f(2, 3) });
    println!("add(100, -5) = {}", unsafe { f(100, -5) });
}
