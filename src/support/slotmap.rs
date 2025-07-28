use std::{
    array::IntoIter,
    marker::PhantomData,
    ops::{Index, IndexMut},
    path::Iter,
};

pub trait Key: Sized + Copy + PartialEq {
    fn new(v: usize) -> Self;

    fn index(&self) -> usize;

    fn none_val() -> Self;

    fn is_none(self) -> bool {
        self == Self::none_val()
    }
}

pub struct PrimaryMap<K: Key, V> {
    values: Vec<Option<V>>,
    _key: PhantomData<K>,
}

impl<K: Key, V> PrimaryMap<K, V> {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            _key: PhantomData,
        }
    }

    pub fn insert(&mut self, val: V) -> K {
        self.values.push(Some(val));
        K::new(self.values.len() - 1)
    }

    pub fn iter(&self) -> PrimaryMapIter<'_, K, V> {
        PrimaryMapIter { map: &self, idx: 0 }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn keys(&self) -> impl Iterator<Item = K> {
        (0..self.values.len()).map(K::new)
    }
}

impl<K: Key, V> Index<K> for PrimaryMap<K, V> {
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        self.values[index.index()].as_ref().unwrap()
    }
}

impl<K: Key, V> IndexMut<K> for PrimaryMap<K, V> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        self.values[index.index()].as_mut().unwrap()
    }
}

pub struct PrimaryMapIter<'i, K: Key, V> {
    map: &'i PrimaryMap<K, V>,
    idx: usize,
}

impl<'i, K: Key, V> Iterator for PrimaryMapIter<'i, K, V> {
    type Item = (K, &'i V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < self.map.values.len() {
            let e = &self.map.values[self.idx];
            let idx = self.idx;
            self.idx += 1;

            if let Some(v) = e.as_ref() {
                return Some((Key::new(idx), v));
            }
        }
        return None;
    }
}

macro_rules! impl_slotmap_key {
    ($type:ty) => {
        impl Key for $type {
            fn new(v: usize) -> Self {
                v as $type
            }

            fn index(&self) -> usize {
                *self as usize
            }

            fn none_val() -> Self {
                <$type>::max_value()
            }
        }
    };
}
impl_slotmap_key!(u8);
impl_slotmap_key!(u16);
impl_slotmap_key!(u32);
impl_slotmap_key!(u64);
impl_slotmap_key!(i8);
impl_slotmap_key!(i16);
impl_slotmap_key!(i32);
impl_slotmap_key!(i64);

#[macro_export]
macro_rules! slotmap_key {
    ($key:ident ($inner_type:ty) ) => {
        #[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Hash, Eq)]
        pub struct $key($inner_type);

        use crate::support::slotmap::Key;

        impl Key for $key {
            fn new(v: usize) -> Self {
                Self(v as $inner_type)
            }

            fn index(&self) -> usize {
                self.0 as usize
            }

            fn none_val() -> Self {
                Self(<$inner_type>::max_value())
            }
        }
    };
}

pub struct SecondaryMap<K: Key, V> {
    values: Vec<V>,
    phantom: PhantomData<K>,
}

impl<K: Key, V: Clone> SecondaryMap<K, V> {
    pub fn new(cap: usize, val: V) -> Self {
        Self {
            values: vec![val.clone(); cap],
            phantom: PhantomData,
        }
    }

    pub fn set(&mut self, key: K, val: V) -> K {
        self.values[key.index()] = val;
        key
    }

    pub fn capacity(&self) -> usize {
        self.values.len()
    }
}

pub trait SecondaryMapExt<K: Key, V: Default + Clone> {
    fn with_default(cap: usize) -> Self;
}

impl<K: Key, V: Default + Clone> SecondaryMapExt<K, V> for SecondaryMap<K, V> {
    fn with_default(cap: usize) -> Self {
        Self::new(cap, Default::default())
    }
}

impl<K: Key, V> IndexMut<K> for SecondaryMap<K, V> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        assert!(index.index() < self.values.len());
        &mut self.values[index.index()]
    }
}

impl<K: Key, V> Index<K> for SecondaryMap<K, V> {
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        assert!(index.index() < self.values.len());
        &self.values[index.index()]
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use super::*;

    slotmap_key!(K(u32));

    impl Debug for K {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "K({})", self.0)
        }
    }

    #[test]
    fn test_primary_map() {
        let mut map = PrimaryMap::new();
        let key: K = map.insert("value");
        assert_eq!(map[key], "value");
    }

    #[test]
    fn test_secondary_map() {
        let mut map = SecondaryMap::with_default(10);
        let key = K::new(0);
        map.set(key, "value");
        assert_eq!(map[key], "value");
    }

    #[test]
    fn test_primary_map_iter() {
        let mut map = PrimaryMap::new();
        map.insert("value1");
        map.insert("value2");

        let mut iter = map.iter();
        assert_eq!(iter.next(), Some((K::new(0), &"value1")));
        assert_eq!(iter.next(), Some((K::new(1), &"value2")));
        assert_eq!(iter.next(), None);
    }
}
