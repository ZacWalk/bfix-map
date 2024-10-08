use std::alloc::{alloc, Layout};
use std::arch::x86_64::{
    __m128i, _mm_cmpeq_epi8, _mm_cvtsi128_si32, _mm_load_si128, _mm_movemask_epi8, _mm_set1_epi8,
};
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher, RandomState};
use std::mem;
use std::ptr::null_mut;
use std::ptr::write;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

fn next_power_of_2_min_256(value: usize) -> usize {
    let value = value.max(256);

    if value.is_power_of_two() {
        return value;
    }

    return 1 << (usize::BITS - value.leading_zeros());
}

const SHARD_BITS: usize = 10;
const SHARD_CAPACITY: usize = 1 << SHARD_BITS;
const SHARD_MASK: u64 = (SHARD_CAPACITY - 1) as u64;
const SHARD_BLOCK_BITS: usize = 4;
const SHARD_BLOCK_SIZE: usize = 1 << SHARD_BLOCK_BITS;
const SHARD_SLOT_MASK: u64 = !0u64 << SHARD_BLOCK_BITS;
const BLOCKS_PER_SHARD: usize = SHARD_CAPACITY / SHARD_BLOCK_SIZE;

#[repr(align(32))]
struct Shard<K: Hash + Eq + Default, V: Default> {
    index: [u8; SHARD_CAPACITY],
    keys: [K; SHARD_CAPACITY],
    values: [V; SHARD_CAPACITY],
}

impl<K: Hash + Eq + Default, V: Default> Shard<K, V> {
    fn new() -> *mut Self {
        unsafe {
            let layout = Layout::new::<Self>();
            let ptr = alloc(layout) as *mut Self;

            let keys_offset = size_of::<[u8; SHARD_CAPACITY]>();
            let values_offset = keys_offset + size_of::<[K; SHARD_CAPACITY]>();

            let index_ptr = (ptr as *mut u8);
            let keys_ptr = (ptr as *mut u8).add(keys_offset) as *mut K;
            let values_ptr = (ptr as *mut u8).add(values_offset) as *mut V;

            for i in 0..SHARD_CAPACITY {
                write(index_ptr.add(i), 0u8);
            }

            for i in 0..SHARD_CAPACITY {
                write(keys_ptr.add(i), K::default());
            }

            for i in 0..SHARD_CAPACITY {
                write(values_ptr.add(i), V::default());
            }

            ptr
        }
    }

    #[inline(always)]
    fn probe_block(&self, block_index: usize, hash8: u8) -> (u32, i32) {
        let match_vec = unsafe { _mm_set1_epi8(hash8 as i8) };
        let index_block = unsafe {
            _mm_load_si128(self.index.as_ptr().offset(block_index as isize) as *const __m128i)
        };
        let found_mask =
            unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(match_vec, index_block)) } as u32;

        let metadata = unsafe { _mm_cvtsi128_si32(index_block) } & 0xFF;

        (found_mask, metadata)
    }

    #[inline(always)]
    pub fn unrolled_search<Q>(&self, bi: usize, found_mask: u32, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let m = found_mask & !0x1;

        // This written in an unrolled way
        // to allow the compiler to optimise it correctly
        let found = m.trailing_zeros();
        if found == u32::BITS {
            return usize::MAX;
        }
        let i = bi + (found as usize);
        let k = unsafe { self.keys.get_unchecked(i) };
        if key == k.borrow() {
            return i;
        }

        let m = m & !(1 << found);
        let found = m.trailing_zeros();
        if found == u32::BITS {
            return usize::MAX;
        }
        let i = bi + (found as usize);
        let k = unsafe { self.keys.get_unchecked(i) };
        if key == k.borrow() {
            return i;
        }

        let m = m & !(1 << found);
        let found = m.trailing_zeros();
        if found == u32::BITS {
            return usize::MAX;
        }
        let i = bi + (found as usize);
        let k = unsafe { self.keys.get_unchecked(i) };
        if key == k.borrow() {
            return i;
        }

        let m = m & !(1 << found);
        let found = m.trailing_zeros();
        if found == u32::BITS {
            return usize::MAX;
        }
        let i = bi + (found as usize);
        let k = unsafe { self.keys.get_unchecked(i) };
        if key == k.borrow() {
            return i;
        }

        let m = m & !(1 << found);
        let found = m.trailing_zeros();
        if found == u32::BITS {
            return usize::MAX;
        }
        let i = bi + (found as usize);
        let k = unsafe { self.keys.get_unchecked(i) };
        if key == k.borrow() {
            return i;
        }

        let m = m & !(1 << found);
        let found = m.trailing_zeros();
        if found == u32::BITS {
            return usize::MAX;
        }
        let i = bi + (found as usize);
        let k = unsafe { self.keys.get_unchecked(i) };
        if key == k.borrow() {
            return i;
        }

        let m = m & !(1 << found);
        let found = m.trailing_zeros();
        if found == u32::BITS {
            return usize::MAX;
        }
        let i = bi + (found as usize);
        let k = unsafe { self.keys.get_unchecked(i) };
        if key == k.borrow() {
            return i;
        }

        let m = m & !(1 << found);
        let found = m.trailing_zeros();
        if found == u32::BITS {
            return usize::MAX;
        }
        let i = bi + (found as usize);
        let k = unsafe { self.keys.get_unchecked(i) };
        if key == k.borrow() {
            return i;
        }

        return usize::MAX;
    }

    #[inline(always)]
    pub fn get<'a, Q>(&'a self, start: usize, hash8: u8, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let mut block_index = start;
        let blocks_per_shard = BLOCKS_PER_SHARD;

        for _ in 0..blocks_per_shard {
            let (found_mask, metadata) = self.probe_block(block_index, hash8);
            let bi = block_index as usize;
            let i = self.unrolled_search(bi, found_mask, key);

            if i != usize::MAX {
                return Some(unsafe { self.values.get_unchecked(i) });
            }

            // not found if no overflow marker
            if metadata != 0xFF {
                break;
            }

            block_index = (block_index + SHARD_BLOCK_SIZE) & SHARD_MASK as usize;
        }

        None // No match found
    }

    #[inline]
    pub fn get_mut<'a, Q>(&'a mut self, start: usize, hash8: u8, key: &Q) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let mut block_index = start;
        let blocks_per_shard = BLOCKS_PER_SHARD;

        for _ in 0..blocks_per_shard {
            let (found_mask, metadata) = self.probe_block(block_index, hash8);
            let bi = block_index as usize;
            let i = self.unrolled_search(bi, found_mask, key);

            if i != usize::MAX {
                return Some(unsafe { self.values.get_unchecked_mut(i) });
            }

            if metadata != 0xFF {
                break;
            }

            block_index = (block_index + SHARD_BLOCK_SIZE) & SHARD_MASK as usize;
        }

        None
    }

    pub fn insert(
        &mut self,
        start: usize,
        hash8: u8,
        key: K,
        value: V,
    ) -> Result<Option<V>, &'static str> {
        let mut block_index = start;
        let blocks_per_shard = BLOCKS_PER_SHARD;

        for _ in 0..blocks_per_shard {
            // look for existing
            let (found_mask, _) = self.probe_block(block_index, hash8);
            let bi = block_index as usize;
            let i = self.unrolled_search(bi, found_mask, &key);

            if i != usize::MAX {
                let v = unsafe { self.values.get_unchecked_mut(i) };
                // Key already exists, replace the value and return the old one
                return Ok(Some(mem::replace(v, value)));
            }

            // look for empty
            let (found_mask, _) = self.probe_block(block_index, 0);
            let m = found_mask & !0x1;
            let found = m.trailing_zeros();
            if found != u32::BITS {
                let i = bi + (found as usize);
                // Found an empty slot
                self.index[i] = hash8;
                unsafe { std::ptr::write(self.keys.as_mut_ptr().add(i), key) };
                unsafe { std::ptr::write(self.values.as_mut_ptr().add(i), value) };
                //self.keys[i] = key;
                //self.values[i] = value;
                return Ok(None);
            }

            // no room - move to next
            self.index[block_index] = 0xFF; // set overflow marker
            block_index = (block_index + SHARD_BLOCK_SIZE) & SHARD_MASK as usize;
        }

        // Shard is full, return an error
        Err("Shard is full")
    }

    pub fn remove<Q>(&mut self, start: usize, hash8: u8, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let mut block_index = start;
        let blocks_per_shard = BLOCKS_PER_SHARD;

        for _ in 0..blocks_per_shard {
            let (found_mask, metadata) = self.probe_block(block_index, hash8);
            let bi = block_index as usize;
            let i = self.unrolled_search(bi, found_mask, key);

            if i != usize::MAX {
                self.index[i] = 0;
                let k = unsafe { self.keys.get_unchecked_mut(i) };
                let v = unsafe { self.values.get_unchecked_mut(i) };
                // let kk = mem::replace(k, K::default());                
                // let vv = mem::replace(v, V::default());

                // Now it's safe to move the values
                let kk = std::mem::replace(k, K::default());
                let vv = std::mem::replace(v, V::default());

                // Set the key to uninitialized memory
                unsafe {
                    std::ptr::drop_in_place(k); // Drop the old key
                    std::ptr::write(k, std::mem::MaybeUninit::uninit().assume_init());

                    std::ptr::drop_in_place(v); // Drop the old key
                    std::ptr::write(v, std::mem::MaybeUninit::uninit().assume_init());
                }
                return Some((kk, vv));
            }

            if metadata != 0xFF {
                break;
            }

            block_index = (block_index + SHARD_BLOCK_SIZE) & SHARD_MASK as usize;
        }

        None
    }
}

const SHARD_EMPTY: usize = 0;
const SHARD_LOCKED: usize = 1;

/// A scoped lock for a shard pointer, ensuring it's unlocked when dropped.
struct shard_lock<'a, K: Hash + Eq + Default, V: Default> {
    shard_ptr_raw: &'a AtomicPtr<Shard<K, V>>,
    original_ptr: *mut Shard<K, V>,
}

impl<'a, K: Hash + Eq + Default, V: Default> shard_lock<'a, K, V> {
    fn new(shard_ptr_raw: &'a AtomicPtr<Shard<K, V>>, original_ptr: *mut Shard<K, V>) -> Self {
        return Self {
            shard_ptr_raw,
            original_ptr,
        };
    }
}

impl<'a, K: Hash + Eq + Default, V: Default> Drop for shard_lock<'a, K, V> {
    fn drop(&mut self) {
        // Restore the original pointer when the lock is dropped
        self.shard_ptr_raw
            .store(self.original_ptr, Ordering::Release);
    }
}

impl<'a, K: Hash + Eq + Default, V: Default> std::ops::Deref for shard_lock<'a, K, V> {
    type Target = Shard<K, V>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.original_ptr }
    }
}

impl<'a, K: Hash + Eq + Default, V: Default> std::ops::DerefMut for shard_lock<'a, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.original_ptr }
    }
}

/// A concurrent hash map with bucket-level fine-grained locking.
///
/// This map is optimized to provide safe concurrent access for multiple threads, allowing
/// simultaneous reads and writes without blocking the entire map. This map uses simd probing.
///
/// Currently the trade-off is that the collection capacity is set at creation. If the
/// collection grows above that limit inserts will fail.
/// I might implement dynamic capacity growth later if needed.
///
/// # Type Parameters
///
/// * `K`: The type of keys stored in the map. Must implement `Hash` and `Eq`.
/// * `V`: The type of values stored in the map.
/// * `S`: The type of build hasher used for hashing keys. Defaults to `RandomState`.
#[repr(align(32))]
pub struct BFixMap<
    K: Hash + Eq + Default,
    V: Default,
    S: BuildHasher + Default + Clone = RandomState,
> {
    //shards: *mut AtomicPtr<Shard<K, V>>,
    shards: Arc<Box<[AtomicPtr<Shard<K, V>>]>>,
    build_hasher: S,
    shard_count: usize,
}

unsafe impl<K, V, S> Send for BFixMap<K, V, S>
where
    K: Hash + Eq + Default + Send,
    V: Default + Send,
    S: BuildHasher + Default + Clone + Send,
    Shard<K, V>: Send,
{
    // No implementation needed, as it's automatically derived if the conditions are met
}

unsafe impl<K, V, S> Sync for BFixMap<K, V, S>
where
    K: Hash + Eq + Default + Sync,
    V: Default + Sync,
    S: BuildHasher + Default + Clone + Sync,
    Shard<K, V>: Sync,
{
    // No implementation needed, as it's automatically derived if the conditions are met
}

impl<K: Hash + Eq + Default, V: Default, S: BuildHasher + Default + Clone> Clone
    for BFixMap<K, V, S>
{
    fn clone(&self) -> Self {
        Self {
            shards: self.shards.clone(),
            build_hasher: self.build_hasher.clone(),
            shard_count: self.shard_count,
        }
    }
}

impl<K: Hash + Eq + Default, V: Default, S: BuildHasher + Default + Clone> BFixMap<K, V, S> {
    pub fn with_capacity_and_hasher(capacity: usize, build_hasher: S) -> Self {
        let shard_count = next_power_of_2_min_256(2 * capacity / SHARD_CAPACITY);
        let shard_ptrs: Vec<_> = (0..shard_count)
            .map(|_| AtomicPtr::new(null_mut()))
            .collect();
        let shards = Arc::new(shard_ptrs.into_boxed_slice());

        Self {
            shards,
            build_hasher,
            shard_count,
        }
    }

    /// Creates a new `BFixMap` with the specified capacity and a default build hasher.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }

    #[inline(always)]
    fn calc_index<Q>(&self, key: &Q) -> (usize, usize, u8)
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        let mut hasher = self.build_hasher.build_hasher();
        key.hash(&mut hasher);
        let h = hasher.finish();
        let shard_mask = (self.shard_count - 1) as u64;
        (
            ((h >> SHARD_BITS) & shard_mask) as usize,
            ((h & SHARD_MASK) & SHARD_SLOT_MASK) as usize,
            ((h & 0xFF).max(1)) as u8,
        )
    }

    /// Retrieves the value associated with the given key from the appropriate shard,
    /// applying the provided reader function to the value if found.
    #[inline(always)]
    pub fn get<Q, R, F>(&self, key: &Q, reader: F) -> Option<R>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
        F: FnOnce(&V) -> R,
    {
        let (shard_index, slot, hash8) = self.calc_index(&key);

        match self.load_shard_ptr(shard_index) {
            Some(shard) => shard.get(slot, hash8, key).map(|v| reader(v)),
            None => None,
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the key already exists, its value is replaced and the old value is returned.
    /// Otherwise, `None` is returned.
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>, &'static str> {
        let (shard_index, slot, hash8) = self.calc_index(&key);

        if let Some(mut shard) = self.load_mut_shard_ptr(shard_index, true) {
            return shard.insert(slot, hash8, key, value);
        }

        Err("Failed to load shard")
    }

    /// Retrieves the value associated with the given key from the appropriate shard,
    /// allowing modification through the provided closure if found.
    pub fn modify<Q, R, F>(&self, key: &Q, mutator: F) -> Option<R>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
        F: FnOnce(&mut V) -> R,
    {
        let (shard_index, slot, hash8) = self.calc_index(key);

        if let Some(mut shard) = self.load_mut_shard_ptr(shard_index, false) {
            if let Some(v) = shard.get_mut(slot, hash8, key) {
                return Some(mutator(v));
            }
        }
        None
    }

    /// Retrieves the value associated with the given key from the appropriate shard,
    /// allowing modification through the provided closure if found.
    pub fn remove<Q>(&self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let (shard_index, slot, hash8) = self.calc_index(key);

        match self.load_mut_shard_ptr(shard_index, false) {
            Some(mut shard) => shard.remove(slot, hash8, key),
            None => None,
        }
    }

    #[inline(always)]
    fn load_shard_ptr(&self, shard_index: usize) -> Option<&Shard<K, V>> {
        loop {
            let shard_ptr_raw = unsafe { self.shards.get_unchecked(shard_index) };
            let shard_ptr = shard_ptr_raw.load(Ordering::Relaxed);
            let shard_ptr_usize = shard_ptr as usize;

            if shard_ptr_usize == SHARD_EMPTY {
                return None;
            } else if shard_ptr_usize == SHARD_LOCKED {
                // Shard locked, spin
                std::thread::yield_now();
                continue;
            } else {
                return Some(unsafe { &mut *shard_ptr });
            }
        }
    }

    fn load_mut_shard_ptr(
        &self,
        shard_index: usize,
        can_create: bool,
    ) -> Option<shard_lock<'_, K, V>> {
        loop {
            let shard_ptr_raw = unsafe { &mut self.shards.get_unchecked(shard_index) };
            let shard_ptr = shard_ptr_raw.load(Ordering::Relaxed);
            let shard_ptr_usize = shard_ptr as usize;

            if !can_create && shard_ptr_usize == SHARD_EMPTY {
                return None;
            };

            if shard_ptr_usize != SHARD_LOCKED {
                let lock_result = shard_ptr_raw.compare_exchange_weak(
                    shard_ptr,
                    SHARD_LOCKED as *mut Shard<K, V>,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );

                if lock_result.is_ok() {
                    // Successfully locked
                    if shard_ptr_usize == SHARD_EMPTY {
                        let new_shard_ptr = Shard::<K, V>::new();

                        // let new_shard_ptr =
                        //     Box::into_raw(Box::new(Shard::<K, V>::new()));
                        return Some(shard_lock::new(shard_ptr_raw, new_shard_ptr));
                    } else {
                        let shard = unsafe { &mut *shard_ptr };
                        return Some(shard_lock::new(shard_ptr_raw, shard_ptr));
                    }
                }
            }

            // Shard locked, spin
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{hash::RandomState, thread};

    use super::*;

    #[test]
    fn test_basic_operations() {
        let map: BFixMap<String, i32, RandomState> =
            BFixMap::<String, i32, RandomState>::with_capacity(10);

        // // Insert
        assert_eq!(map.insert("one".to_string(), 1).unwrap(), None);
        assert_eq!(map.insert("two".to_string(), 2).unwrap(), None);
        assert_eq!(map.insert("x".to_string(), 3).unwrap(), None);

        // Get
        assert_eq!(map.get("one", |v| v.clone()), Some(1));
        assert_eq!(map.get("x", |v| v.clone()), Some(3));
        assert_eq!(map.get(&"two".to_string(), |v| v.clone()), Some(2));
        assert_eq!(map.get("three", |v| v.clone()), None);

        // Modify
        assert_eq!(
            map.modify("one", |v| {
                *v += 1;
                *v
            }),
            Some(2)
        );
        assert_eq!(map.get("one", |v| v.clone()), Some(2));
        assert_eq!(
            map.modify("three", |v| {
                *v += 1;
                *v
            }),
            None
        );

        assert_eq!(map.remove("one"), Some(("one".to_string(), 2)));
        assert_eq!(map.remove("two"), Some(("two".to_string(), 2)));
        assert_eq!(map.get("one", |v| v.clone()), None);

        // Insert into deallocated
        assert_eq!(map.insert("three".to_string(), 11).unwrap(), None);
        assert_eq!(map.insert("four".to_string(), 22).unwrap(), None);

        assert_eq!(map.get("three", |v| v.clone()), Some(11));
        assert_eq!(map.get("four", |v| v.clone()), Some(22));
    }

    #[test]
    fn test_large_capacity() {
        const CAPACITY: usize = 1_000_000;

        let map = BFixMap::<usize, usize, ahash::RandomState>::with_capacity_and_hasher(
            CAPACITY,
            ahash::RandomState::new(),
        );

        // Insert a large number of items
        for i in 0..CAPACITY {
            map.insert(i, i);
        }

        // Verify all items are present
        for i in 0..CAPACITY {
            assert_eq!(map.get(&i, |v| v.clone()), Some(i));
        }
    }

    #[test]
    fn test_multithreaded_access() {
        const NUM_THREADS: usize = 100;
        const NUM_KEYS_PER_THREAD: usize = 10000;

        let map = BFixMap::<usize, usize, RandomState>::with_capacity_and_hasher(
            NUM_THREADS * NUM_KEYS_PER_THREAD,
            RandomState::default(),
        );

        let mut handles = vec![];

        for thread_id in 0..NUM_THREADS {
            let map_clone = map.clone();
            let handle = thread::spawn(move || {
                for i in 0..NUM_KEYS_PER_THREAD {
                    let key = thread_id * NUM_KEYS_PER_THREAD + i;
                    map_clone.insert(key, key);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all keys and values are present
        for thread_id in 0..NUM_THREADS {
            for i in 0..NUM_KEYS_PER_THREAD {
                let key = thread_id * NUM_KEYS_PER_THREAD + i;
                assert_eq!(map.get(&key, |v| v.clone()), Some(key));
            }
        }
    }
}
