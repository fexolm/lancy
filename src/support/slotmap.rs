use std::{
    array::IntoIter,
    marker::PhantomData,
    ops::{Index, IndexMut},
    path::Iter,
};

pub trait Key: Sized + Copy + PartialEq + Default {
    fn new(v: usize) -> Self;

    fn index(&self) -> usize;

    fn none_val() -> Self;

    fn is_none(self) -> bool {
        self == Self::none_val()
    }
}

pub struct PrimaryMap<K: Key, V> {
    values: Vec<Option<V>>,
    freelist: Vec<K>,
}

impl<K: Key, V> PrimaryMap<K, V> {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            freelist: Vec::new(),
        }
    }

    pub fn insert(&mut self, val: V) -> K {
        if let Some(key) = self.freelist.pop() {
            self.values[key.index()] = Some(val);
            key
        } else {
            self.values.push(Some(val));
            K::new(self.values.len() - 1)
        }
    }

    pub fn iter(&self) -> PrimaryMapIter<'_, K, V> {
        PrimaryMapIter { map: &self, idx: 0 }
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

#[macro_export]
macro_rules! slotmap_key {
    ($key:ident ($inner_type:ty) ) => {
        use crate::support::slotmap::Key;
        #[derive(Clone, Copy, PartialEq)]
        pub struct $key($inner_type);

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

        impl Default for $key {
            fn default() -> Self {
                Self::none_val()
            }
        }
    };
}

pub struct SecondaryMap<K: Key, V> {
    values: Vec<V>,
    phantom: PhantomData<K>,
}

impl<K: Key, V: Default> SecondaryMap<K, V> {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            phantom: PhantomData,
        }
    }

    pub fn insert(&mut self, key: K, val: V) -> K {
        if self.values.len() <= key.index() {
            self.values.resize_with(key.index() + 1, Default::default);
        }
        self.values[key.index()] = val;
        key
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
