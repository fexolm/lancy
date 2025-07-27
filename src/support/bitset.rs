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

    pub fn add(&mut self, index: usize) {
        assert!(index < self.buckets.len() * Self::bits_in_bucket());
        let num_bucket = index / Self::bits_in_bucket();
        let bit_pos = index % (Self::bits_in_bucket());
        self.buckets[num_bucket] |= 1 << bit_pos;
    }

    pub fn del(&mut self, index: usize) {
        assert!(index < self.buckets.len() * Self::bits_in_bucket());
        let num_bucket = index / Self::bits_in_bucket();
        let bit_pos = index % (Self::bits_in_bucket());
        self.buckets[num_bucket] &= !(1 << bit_pos);
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

    pub fn has(&self, index: usize) -> bool {
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
        assert_eq!(bs.buckets.len(), (100 + 31) / 32);
    }

    #[test]
    fn test_add_and_has() {
        let mut bs = FixedBitSet::new(64);
        bs.add(0);
        bs.add(31);
        bs.add(32);
        bs.add(63);
        assert!(bs.has(0));
        assert!(bs.has(31));
        assert!(bs.has(32));
        assert!(bs.has(63));
        assert!(!bs.has(1));
        assert!(!bs.has(62));
        assert!(!bs.has(64));
    }

    #[test]
    fn test_del() {
        let mut bs = FixedBitSet::new(40);
        bs.add(10);
        assert!(bs.has(10));
        bs.del(10);
        assert!(!bs.has(10));
    }

    #[test]
    fn test_set_true_and_false() {
        let mut bs = FixedBitSet::new(10);
        bs.set(true, 5);
        assert!(bs.has(5));
        bs.set(false, 5);
        assert!(!bs.has(5));
    }

    #[test]
    fn test_union() {
        let mut a = FixedBitSet::new(64);
        let mut b = FixedBitSet::new(64);
        a.add(1);
        a.add(2);
        b.add(2);
        b.add(3);
        a.union(&b);
        assert!(a.has(1));
        assert!(a.has(2));
        assert!(a.has(3));
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn test_intersect() {
        let mut a = FixedBitSet::new(64);
        let mut b = FixedBitSet::new(64);
        a.add(1);
        a.add(2);
        b.add(2);
        b.add(3);
        a.intersect(&b);
        assert!(!a.has(1));
        assert!(a.has(2));
        assert!(!a.has(3));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn test_equals() {
        let mut a = FixedBitSet::new(32);
        let mut b = FixedBitSet::new(32);
        assert!(a.equals(&b));
        a.add(5);
        assert!(!a.equals(&b));
        b.add(5);
        assert!(a.equals(&b));
        a.add(10);
        b.add(11);
        assert!(!a.equals(&b));
    }

    #[test]
    fn test_set_resize() {
        let mut bs = FixedBitSet::new(1);
        bs.set(true, 100);
        assert!(bs.has(100));
        assert_eq!(bs.buckets.len(), (100 / 32) + 1);
    }
}
