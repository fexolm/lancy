#[derive(Clone, Copy)]
pub enum RegClass {
    Int(u8),
    Float(u8),
    Vec(u8),
}

#[derive(Clone, Copy)]
pub struct Reg {
    repr: u32,
}

impl Reg {
    pub fn virt(_cls: RegClass, id: u32) -> Self {
        Reg { repr: id }
    }

    pub fn is_virtual(&self) -> bool {
        true
    }

    pub fn is_fixed(&self) -> bool {
        !self.is_virtual()
    }

    pub fn class(&self) -> RegClass {
        todo!();
    }

    pub fn id(&self) -> u32 {
        self.repr
    }
}
