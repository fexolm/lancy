use core::fmt;
use std::{
    fmt::{Display, Formatter},
    mem::size_of,
};

use smallvec::SmallVec;

type Word = u32;

#[derive(Clone)]
pub struct FixedBitSet {
    buckets: SmallVec<[Word; 4]>,
}

impl FixedBitSet {
    pub fn new(size: usize) -> Self {
        use std::cmp::max;
        let words = (size + Self::bits_in_bucket() - 1) / Self::bits_in_bucket();
        let mut buckets = SmallVec::with_capacity(words);
        buckets.resize(words, 0);
        Self { buckets }
    }

    fn bits_in_bucket() -> usize {
        return size_of::<Word>() * 8;
    }

    pub fn len(&self) -> usize {
        self.buckets.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn intersect(&mut self, other: &FixedBitSet) {
        debug_assert_eq!(self.buckets.len(), other.buckets.len());
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            *bucket &= other.buckets[i];
        }
    }

    pub fn union(&mut self, other: &FixedBitSet) {
        debug_assert_eq!(self.buckets.len(), other.buckets.len());
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            *bucket |= other.buckets[i];
        }
    }

    pub fn set(&mut self, val: bool, index: usize) {
        let num_bucket = index / Self::bits_in_bucket();
        if num_bucket + 1 > self.buckets.len() {
            self.buckets.resize(num_bucket + 1, 0);
        }
        let bit_pos = index % (Self::bits_in_bucket());
        if val {
            self.buckets[num_bucket] |= 1 << bit_pos;
        } else {
            self.buckets[num_bucket] &= !(1 << bit_pos);
        }
    }

    pub fn get(&self, index: usize) -> bool {
        if index >= self.buckets.len() * 32 {
            return false;
        }
        let num_bucket = index / Self::bits_in_bucket();
        let bit_pos = index % Self::bits_in_bucket();
        return self.buckets[num_bucket] & (1 << bit_pos) != 0;
    }

    pub fn equals(&self, other: &FixedBitSet) -> bool {
        if self.buckets.len() != other.buckets.len() {
            return false;
        }
        for (a, b) in self.buckets.iter().zip(other.buckets.iter()) {
            if a != b {
                return false;
            }
        }
        true
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_len() {
        let bs = FixedBitSet::new(100);
        assert_eq!(bs.len(), 0);
        // All bits should be unset
        for i in 0..100 {
            assert!(!bs.get(i));
        }
    }

    #[test]
    fn test_set_and_get() {
        let mut bs = FixedBitSet::new(10);
        bs.set(true, 3);
        bs.set(true, 7);
        assert!(bs.get(3));
        assert!(bs.get(7));
        assert!(!bs.get(0));
        assert!(!bs.get(9));
        assert_eq!(bs.len(), 2);

        bs.set(false, 3);
        assert!(!bs.get(3));
        assert_eq!(bs.len(), 1);
    }

    #[test]
    fn test_set_out_of_bounds() {
        let mut bs = FixedBitSet::new(5);
        bs.set(true, 31);
        assert!(bs.get(31));
        assert_eq!(bs.len(), 1);
        bs.set(true, 32);
        assert!(bs.get(32));
        assert_eq!(bs.len(), 2);
    }

    #[test]
    fn test_intersect() {
        let mut a = FixedBitSet::new(10);
        let mut b = FixedBitSet::new(10);
        a.set(true, 1);
        a.set(true, 2);
        b.set(true, 2);
        b.set(true, 3);
        a.intersect(&b);
        assert!(!a.get(1));
        assert!(a.get(2));
        assert!(!a.get(3));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn test_union() {
        let mut a = FixedBitSet::new(10);
        let mut b = FixedBitSet::new(10);
        a.set(true, 1);
        b.set(true, 2);
        a.union(&b);
        assert!(a.get(1));
        assert!(a.get(2));
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let bs = FixedBitSet::new(10);
        assert!(!bs.get(1000));
    }
    #[test]
    fn test_large_set_basic() {
        let size = 10_000;
        let mut bs = FixedBitSet::new(size);
        // Set every 100th bit
        for i in (0..size).step_by(100) {
            bs.set(true, i);
        }
        // Check set bits
        for i in 0..size {
            if i % 100 == 0 {
                assert!(bs.get(i), "Bit {} should be set", i);
            } else {
                assert!(!bs.get(i), "Bit {} should not be set", i);
            }
        }
        assert_eq!(bs.len(), size / 100);
    }

    #[test]
    fn test_large_set_union_and_intersect() {
        let size = 5000;
        let mut a = FixedBitSet::new(size);
        let mut b = FixedBitSet::new(size);

        // a: set even bits, b: set bits divisible by 3
        for i in (0..size).step_by(2) {
            a.set(true, i);
        }
        for i in (0..size).step_by(3) {
            b.set(true, i);
        }

        let mut c = a.clone();
        c.union(&b);
        for i in 0..size {
            if i % 2 == 0 || i % 3 == 0 {
                assert!(c.get(i), "Bit {} should be set in union", i);
            } else {
                assert!(!c.get(i), "Bit {} should not be set in union", i);
            }
        }

        let mut d = a.clone();
        d.intersect(&b);
        for i in 0..size {
            if i % 2 == 0 && i % 3 == 0 {
                assert!(d.get(i), "Bit {} should be set in intersection", i);
            } else {
                assert!(!d.get(i), "Bit {} should not be set in intersection", i);
            }
        }
    }

    #[test]
    fn test_large_set_set_and_unset() {
        let size = 2048;
        let mut bs = FixedBitSet::new(size);
        // Set all bits
        for i in 0..size {
            bs.set(true, i);
        }
        assert_eq!(bs.len(), size);

        // Unset all bits
        for i in 0..size {
            bs.set(false, i);
        }
        assert_eq!(bs.len(), 0);
        for i in 0..size {
            assert!(!bs.get(i));
        }
    }

    #[test]
    fn test_large_set_out_of_bounds() {
        let size = 1000;
        let bs = FixedBitSet::new(size);
        assert!(!bs.get(size * 10));
        assert!(!bs.get(usize::MAX));
    }

    #[test]
    fn test_equals_basic() {
        let mut a = FixedBitSet::new(16);
        let mut b = FixedBitSet::new(16);
        assert!(a.equals(&b));
        a.set(true, 5);
        assert!(!a.equals(&b));
        b.set(true, 5);
        assert!(a.equals(&b));
        a.set(true, 10);
        b.set(true, 11);
        assert!(!a.equals(&b));
    }
}
