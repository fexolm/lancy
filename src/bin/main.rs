pub enum GenericInstruction {
    BR,
}

pub enum Inst<I: Sized> {
    Generic(GenericInstruction),
    Target(I),
}

pub enum X64Inst {
    MOV,
}


fn main() {
    println!("Hello, world!");

    println!("{}", std::mem::size_of::<Inst<X64Inst>>());
}
