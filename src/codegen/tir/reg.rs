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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg {
    repr: u32,
}

impl Reg {
    pub const fn new(typ: RegType, cls: RegClass, id: u32) -> Self {
        Reg {
            repr: id
                | Self::get_size_mask(cls)
                | Self::get_class_mask(cls)
                | Self::get_type_mask(typ),
        }
    }

    pub fn get_type(self) -> RegType {
        if self.repr & VIRT_BIT != 0 {
            RegType::Virtual
        } else if self.repr & SPILL_BIT != 0 {
            RegType::Spill
        } else {
            RegType::Physical
        }
    }

    pub fn get_class(self) -> RegClass {
        let size = 1 << self.reg_class_size();

        if self.repr & REG_CLASS_INT_BIT != 0 {
            RegClass::Int(size)
        } else if self.repr & REG_CLASS_FLOAT_BIT != 0 {
            RegClass::Float(size)
        } else {
            RegClass::Vec(size)
        }
    }

    pub fn get_id(self) -> u32 {
        self.repr & ID_MASK
    }

    fn reg_class_size(self) -> u8 {
        ((self.repr & REG_CLASS_MASK) >> REG_CLASS_SIZE_OFFSET) as u8
    }

    const fn get_size_mask(class: RegClass) -> u32 {
        let sz = match class {
            RegClass::Int(s) => s,
            RegClass::Float(s) => s,
            RegClass::Vec(s) => s,
        };

        (((sz - 1).count_ones()) as u32) << REG_CLASS_SIZE_OFFSET
    }

    const fn get_class_mask(class: RegClass) -> u32 {
        match class {
            RegClass::Int(_) => REG_CLASS_INT_BIT,
            RegClass::Float(_) => REG_CLASS_FLOAT_BIT,
            RegClass::Vec(_) => 0,
        }
    }

    const fn get_type_mask(typ: RegType) -> u32 {
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
            assert_eq!(reg.get_type(), typ);
            assert_eq!(reg.get_class(), cls);
            assert_eq!(reg.get_id(), id);
        }
        test_case(RegType::Physical, RegClass::Float(8), 12);
        test_case(RegType::Virtual, RegClass::Int(4), 48);
        test_case(RegType::Spill, RegClass::Vec(32), 61);
    }
}
#[test]
fn test_reg_edge_cases() {
    // Test minimum id
    let reg = Reg::new(RegType::Physical, RegClass::Int(1), 0);
    assert_eq!(reg.get_type(), RegType::Physical);
    assert_eq!(reg.get_class(), RegClass::Int(1));
    assert_eq!(reg.get_id(), 0);

    // Test maximum id
    let max_id = ID_MASK;
    let reg = Reg::new(RegType::Virtual, RegClass::Vec(16), max_id);
    assert_eq!(reg.get_type(), RegType::Virtual);
    assert_eq!(reg.get_class(), RegClass::Vec(16));
    assert_eq!(reg.get_id(), max_id);

    // Test all RegType variants with same class and id
    let id = 7;
    let class = RegClass::Float(2);
    let reg_phys = Reg::new(RegType::Physical, class, id);
    let reg_virt = Reg::new(RegType::Virtual, class, id);
    let reg_spill = Reg::new(RegType::Spill, class, id);
    assert_eq!(reg_phys.get_type(), RegType::Physical);
    assert_eq!(reg_virt.get_type(), RegType::Virtual);
    assert_eq!(reg_spill.get_type(), RegType::Spill);

    // Test all RegClass variants with same type and id
    let id = 3;
    let reg_int = Reg::new(RegType::Physical, RegClass::Int(8), id);
    let reg_float = Reg::new(RegType::Physical, RegClass::Float(8), id);
    let reg_vec = Reg::new(RegType::Physical, RegClass::Vec(8), id);
    assert_eq!(reg_int.get_class(), RegClass::Int(8));
    assert_eq!(reg_float.get_class(), RegClass::Float(8));
    assert_eq!(reg_vec.get_class(), RegClass::Vec(8));
}

#[test]
#[should_panic]
fn test_invalid_size_non_power_of_two() {
    // Should panic because 3 is not a power of two
    let _ = Reg::new(RegType::Physical, RegClass::Int(3), 1);
}

#[test]
fn test_reg_class_size_encoding() {
    // Test that sizes are encoded/decoded correctly for powers of two
    for &size in &[1u8, 2, 4, 8, 16, 32, 64, 128] {
        let reg = Reg::new(RegType::Physical, RegClass::Vec(size), 5);
        assert_eq!(reg.get_class(), RegClass::Vec(size));
    }
}
