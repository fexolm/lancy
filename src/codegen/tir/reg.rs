#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RegClass {
    Int(u8),
    Float(u8),
    Vec(u8),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RegType {
    Virtual,
    Physical,
    Spill,
}

const VIRT_BIT: u32 = 1 << 31;
const SPILL_BIT: u32 = 1 << 30;
const REG_CLASS_INT_BIT: u32 = 1 << 29;
const REG_CLASS_FLOAT_BIT: u32 = 1 << 28;
const REG_CLASS_SIZE_BITS: u32 = 4;
const REG_CLASS_SIZE_OFFSET: u32 = 28 - REG_CLASS_SIZE_BITS;
const REG_CLASS_MASK: u32 = ((1 << REG_CLASS_SIZE_BITS) - 1) << REG_CLASS_SIZE_OFFSET;
const ID_MASK: u32 = (1 << REG_CLASS_SIZE_OFFSET) - 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reg {
    repr: u32,
}

impl Reg {
    pub fn new(typ: RegType, cls: RegClass, id: u32) -> Self {
        debug_assert_eq!(id & ID_MASK, id);

        Reg {
            repr: id
                | Self::get_size_mask(cls)
                | Self::get_class_mask(cls)
                | Self::get_type_mask(typ),
        }
    }

    pub fn typ(self) -> RegType {
        if self.repr & VIRT_BIT != 0 {
            RegType::Virtual
        } else if self.repr & SPILL_BIT != 0 {
            RegType::Spill
        } else {
            RegType::Physical
        }
    }

    pub fn class(self) -> RegClass {
        let size = 1 << self.reg_class_size();

        if self.repr & REG_CLASS_INT_BIT != 0 {
            RegClass::Int(size)
        } else if self.repr & REG_CLASS_FLOAT_BIT != 0 {
            RegClass::Float(size)
        } else {
            RegClass::Vec(size)
        }
    }

    pub fn id(self) -> u32 {
        self.repr & ID_MASK
    }

    fn reg_class_size(self) -> u8 {
        ((self.repr & REG_CLASS_MASK) >> REG_CLASS_SIZE_OFFSET) as u8
    }

    fn get_size_mask(class: RegClass) -> u32 {
        let sz = match class {
            RegClass::Int(s) => s,
            RegClass::Float(s) => s,
            RegClass::Vec(s) => s,
        };

        debug_assert_eq!(sz & (sz - 1), 0);

        (((sz - 1).count_ones()) as u32) << REG_CLASS_SIZE_OFFSET
    }

    fn get_class_mask(class: RegClass) -> u32 {
        match class {
            RegClass::Int(_) => REG_CLASS_INT_BIT,
            RegClass::Float(_) => REG_CLASS_FLOAT_BIT,
            RegClass::Vec(_) => 0,
        }
    }

    fn get_type_mask(typ: RegType) -> u32 {
        match typ {
            RegType::Virtual => VIRT_BIT,
            RegType::Physical => 0,
            RegType::Spill => SPILL_BIT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reg() {
        fn test_case(typ: RegType, cls: RegClass, id: u32) {
            let reg = Reg::new(typ, cls, id);
            assert_eq!(reg.typ(), typ);
            assert_eq!(reg.class(), cls);
            assert_eq!(reg.id(), id);
        }
        test_case(RegType::Physical, RegClass::Float(8), 12);
        test_case(RegType::Virtual, RegClass::Int(4), 48);
        test_case(RegType::Spill, RegClass::Vec(32), 61);
    }
}
