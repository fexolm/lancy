use std::{
    marker::PhantomData,
    ops::{Index, IndexMut}
    ,
};

pub trait Key: Sized + Copy + PartialEq {
    const NONE_VAL: Self;
    fn new(v: usize) -> Self;

    fn index(&self) -> usize;

    fn is_none(self) -> bool {
        self == Self::NONE_VAL
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

    pub fn keys(&self) -> impl Iterator<Item=K> {
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
        None
    }
}

macro_rules! impl_slotmap_key {
    ($type:ty) => {
        impl Key for $type {
            const NONE_VAL: Self = <$type>::MAX;

            fn new(v: usize) -> Self {
                v as $type
            }

            fn index(&self) -> usize {
                *self as usize
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
        #[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Hash, Eq, Default)]
        pub struct $key(pub $inner_type);

        use crate::support::slotmap::Key;

        impl Key for $key {
            const NONE_VAL: Self = Self(<$inner_type>::MAX);

            fn new(v: usize) -> Self {
                Self(v as $inner_type)
            }

            fn index(&self) -> usize {
                self.0 as usize
            }
        }
    };
}

#[derive(Default)]
pub struct SecondaryMap<K: Key, V> {
    values: Vec<Option<V>>,
    phantom: PhantomData<K>,
}

impl<K: Key, V: Clone> SecondaryMap<K, V> {
    pub fn new(cap: usize) -> Self {
        Self {
            values: vec![None; cap],
            phantom: PhantomData,
        }
    }

    pub fn fill(&mut self, val: V) {
        for v in self.values.iter_mut() {
            *v = Some(val.clone());
        }
    }

    pub fn add(&mut self, key: K, val: V) -> K {
        self.values[key.index()] = Some(val);
        key
    }

    pub fn contains(&self, key: K) -> bool {
        key.index() < self.values.len() && self.values[key.index()].is_some()
    }

    pub fn get(&self, key: K) -> Option<&V> {
        if key.index() < self.values.len() {
            self.values[key.index()].as_ref()
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        if key.index() < self.values.len() {
            self.values[key.index()].as_mut()
        } else {
            None
        }
    }

    pub fn capacity(&self) -> usize {
        self.values.len()
    }

    pub fn iter(&self) -> impl Iterator<Item=(K, &V)> {
        self.values.iter().enumerate().filter_map(|(i, v)| {
            if let Some(v) = v.as_ref() {
                Some((K::new(i), v))
            } else {
                None
            }
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=(K, &mut V)> {
        self.values.iter_mut().enumerate().filter_map(|(i, v)| {
            if let Some(v) = v.as_mut() {
                Some((K::new(i), v))
            } else {
                None
            }
        })
    }

    pub fn keys(&self) -> impl Iterator<Item=K> {
        (0..self.values.len()).map(K::new)
    }

    pub fn values(&self) -> impl Iterator<Item=&V> {
        self.values.iter().flatten()
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item=&mut V> {
        self.values.iter_mut().flatten()
    }
}

impl<K: Key, V: Default> IndexMut<K> for SecondaryMap<K, V> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        assert!(index.index() < self.values.len());
        if self.values[index.index()].is_none() {
            self.values[index.index()] = Some(Default::default());
        }
        self.values[index.index()].as_mut().unwrap()
    }
}


impl<K: Key, V> Index<K> for SecondaryMap<K, V> {
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        assert!(index.index() < self.values.len());
        self.values[index.index()].as_ref().unwrap()
    }
}

pub struct SecondaryMapIter<'a, K: Key, V> {
    map: &'a SecondaryMap<K, V>,
    idx: usize,
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
        let mut map = SecondaryMap::new(10);
        let key = K::new(0);
        map.add(key, "value");
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

    #[test]
    fn test_secondary_map_iter() {
        let mut map = SecondaryMap::new(10);
        map.add(K::new(0), "value");

        let mut iter = map.iter();
        assert_eq!(iter.next(), Some((K::new(0), &"value")));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_secondary_map_iter_mut() {
        let mut map = SecondaryMap::new(10);
        let key = K::new(0);
        map.add(key, "value");

        let mut iter = map.iter_mut();
        assert_eq!(iter.next(), Some((K::new(0), &mut "value")));
        assert_eq!(iter.next(), None);
    }
}
