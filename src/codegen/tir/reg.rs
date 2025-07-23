pub enum RegClass {
    Int(u8),
    Float(u8),
    Vec(u8),
}

pub struct Reg {
    repr: u32,
}

impl Reg {
    pub fn is_virtual(&self) -> bool {
        todo!();
    }

    pub fn is_fixed(&self) -> bool {
        !self.is_virtual()
    }

    pub fn class(&self) -> RegClass {
        todo!();
    }
}