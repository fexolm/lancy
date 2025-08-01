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
    fn new(size: usize, value: Word) -> Self {
        use std::cmp::max;
        let words = (size + Self::bits_in_bucket() - 1) / Self::bits_in_bucket();
        let mut buckets = SmallVec::with_capacity(words);
        buckets.resize(words, value);
        Self { buckets }
    }

    pub fn zeroes(size: usize) -> Self {
        Self::new(size, 0)
    }

    pub fn ones(size: usize) -> Self {
        Self::new(size, u32::MAX)
    }

    fn bits_in_bucket() -> usize {
        return size_of::<Word>() * 8;
    }

    pub fn ones_count(&self) -> usize {
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

    pub fn difference(&mut self, other: &FixedBitSet) {
        debug_assert_eq!(self.buckets.len(), other.buckets.len());
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            *bucket &= !other.buckets[i];
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

    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.buckets.iter().enumerate().flat_map(|(i, &bucket)| {
            (0..Self::bits_in_bucket())
                .filter(move |j| bucket & (1 << j) != 0)
                .map(move |j| i * Self::bits_in_bucket() + j)
        })
    }

    pub fn iter_zeroes(&self) -> impl Iterator<Item = usize> + '_ {
        self.buckets.iter().enumerate().flat_map(|(i, &bucket)| {
            (0..Self::bits_in_bucket())
                .filter(move |j| bucket & (1 << j) == 0)
                .map(move |j| i * Self::bits_in_bucket() + j)
        })
    }

    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = 0;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_len() {
        let bs = FixedBitSet::zeroes(100);
        assert_eq!(bs.ones_count(), 0);
        assert_eq!(bs.buckets.len(), (100 + 31) / 32);
    }

    #[test]
    fn test_add_and_has() {
        let mut bs = FixedBitSet::zeroes(64);
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
        let mut bs = FixedBitSet::zeroes(40);
        bs.add(10);
        assert!(bs.has(10));
        bs.del(10);
        assert!(!bs.has(10));
    }

    #[test]
    fn test_union() {
        let mut a = FixedBitSet::zeroes(64);
        let mut b = FixedBitSet::zeroes(64);
        a.add(1);
        a.add(2);
        b.add(2);
        b.add(3);
        a.union(&b);
        assert!(a.has(1));
        assert!(a.has(2));
        assert!(a.has(3));
        assert_eq!(a.ones_count(), 3);
    }

    #[test]
    fn test_intersect() {
        let mut a = FixedBitSet::zeroes(64);
        let mut b = FixedBitSet::zeroes(64);
        a.add(1);
        a.add(2);
        b.add(2);
        b.add(3);
        a.intersect(&b);
        assert!(!a.has(1));
        assert!(a.has(2));
        assert!(!a.has(3));
        assert_eq!(a.ones_count(), 1);
    }

    #[test]
    fn test_equals() {
        let mut a = FixedBitSet::zeroes(32);
        let mut b = FixedBitSet::zeroes(32);
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
    fn test_iter_ones() {
        let mut bs = FixedBitSet::zeroes(64);
        bs.add(1);
        bs.add(3);
        bs.add(32);
        let ones: Vec<usize> = bs.iter_ones().collect();
        assert_eq!(ones, vec![1, 3, 32]);
    }
}
