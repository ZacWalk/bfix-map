use std::arch::x86_64::*;
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher, RandomState};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU16, AtomicUsize, Ordering};
use std::{array, mem};

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
const SHARD_PROBE_BLOCK: usize = 16;
const SHARD_SLOT_MASK: u64 = !0u64 << 4;

#[repr(align(32))]
struct Shard<K: Hash + Eq + Default + Clone, V: Clone + Default> {
    index: [u8; SHARD_CAPACITY],
    data: [(K, V); SHARD_CAPACITY],
}

impl<K: Hash + Eq + Default + Clone, V: Clone + Default> Shard<K, V> {
    fn new() -> Self {
        Self {
            index: [0; SHARD_CAPACITY],
            data: array::from_fn(|_| (K::default(), V::default())),
        }
    }

    #[inline]
    pub fn get<Q>(&self, start: usize, kh: u8, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        unsafe {
            let mut block_index = start;
            //let match_vec = _mm256_set1_epi8(kh as i8);
            let match_vec = _mm_set1_epi8(kh as i8);

            debug_assert!(self.index.as_ptr() as usize % 32 == 0, "Index not alligned to 32 bytes {:x}", self.index.as_ptr() as usize);

            for _ in 0..SHARD_CAPACITY / SHARD_PROBE_BLOCK {
                // let entries = _mm256_load_si256(self.index.as_ptr().offset(min as isize) as *const __m256i);
                // let found_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(match_vec, entries));

                let entries = _mm_load_si128(self.index.as_ptr().offset(block_index as isize) as *const __m128i);
                let found_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(match_vec, entries));

                for i in 1..SHARD_PROBE_BLOCK {
                    // bits set represent a possible match
                    if (found_mask & (1 << i)) != 0 {
                        if self.data[block_index + i].0.borrow() == key {
                            return Some(self.data[block_index + i].1.clone());
                        }
                    }
                }

                // let max = min + SHARD_PROBE_BLOCK;

                // for i in min + 1..max {
                //     if self.index[i] == kh {
                //         // Potential match in the index, check the key in data
                //         if self.data[i].0.borrow() == key {
                //             return Some(self.data[i].1.clone()); // Return a clone of the value
                //         }
                //     }
                // }

                // no overflow marker
                if self.index[block_index] != 0xFF                
                {
                    break;
                }

                block_index = (block_index + SHARD_PROBE_BLOCK) & SHARD_MASK as usize;
            }

            None // No match found
        }
    }

    pub fn insert(&mut self, start: usize, kh: u8, key: K, value: V) -> Option<V> {
        let mut block_start = start;

        for _ in 0..SHARD_CAPACITY / SHARD_PROBE_BLOCK {
            if self.index[block_start] != 0xFF {
                let block_end = block_start + SHARD_PROBE_BLOCK;

                for i in block_start + 1..block_end {
                    if self.index[i] == 0 {
                        // Found an empty slot
                        self.index[i] = kh;
                        self.data[i] = (key, value);
                        return None;
                    } else if self.index[i] == kh && self.data[i].0 == key {
                        // Key already exists, replace the value and return the old one
                        return Some(mem::replace(&mut self.data[i].1, value));
                    }
                }

                // no room - move to next
                self.index[block_start] = 0xFF; // set overflow marker                
            }
            block_start = (block_start + SHARD_PROBE_BLOCK) & SHARD_MASK as usize;
        }

        None
    }
}

const SHARD_EMPTY: usize = 0;
const SHARD_LOCKED: usize = 1;


/// A concurrent hash map with bucket-level fine-grained locking.
///
/// This map is optimized to provide safe concurrent access for multiple threads, allowing
/// simultaneous reads and writes without blocking the entire map.
///
/// This map has a naive implementation however it turns out to have good performance
/// with large numbers of threads. The trade-off is that the max number of shards is set at
/// creation time based on the provided capacity. The collection can grow to contain larger
/// numbers of items than the specified capacity, but the number of shards does not change.
/// This design avoids any complex mechanisms around splitting shards.
///
/// # Type Parameters
///
/// * `K`: The type of keys stored in the map. Must implement `Hash` and `Eq`.
/// * `V`: The type of values stored in the map. Must implement `Clone`.
/// * `S`: The type of build hasher used for hashing keys. Defaults to `RandomState`.
pub struct BFixMap<
    K: Hash + Eq + Default + Clone,
    V: Clone + Default,
    S: BuildHasher + Default = RandomState,
> {
    shards: Vec<AtomicPtr<Shard<K, V>>>,
    build_hasher: S,
    shard_count: usize,
}

impl<K: Hash + Eq + Default + Clone, V: Clone + Default, S: BuildHasher + Default>
    BFixMap<K, V, S>
{
    pub fn with_capacity_and_hasher(capacity: usize, build_hasher: S) -> Self {
        let shard_count = next_power_of_2_min_256(capacity / (SHARD_CAPACITY / 2));
        let shards: Vec<AtomicPtr<Shard<K, V>>> = (0..shard_count)
            .map(|_| AtomicPtr::new(null_mut()))
            .collect();

        Self {
            shards: shards,
            build_hasher,
            shard_count,
        }
    }

    /// Creates a new `BFixMap` with the specified capacity and a default build hasher.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }

    #[inline]
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
            ((h & 0xFF) | 0x80) as u8,
        )
    }

    /// Retrieves the value associated with the given key, if it exists.
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let (shard_index, slot, kh) = self.calc_index(&key);

        loop {
            let shard_ptr = self.shards[shard_index].load(Ordering::Relaxed);
            let shard_ptr_usize = shard_ptr as usize;

            if shard_ptr_usize == SHARD_EMPTY {
                return None;
            } else if shard_ptr_usize == SHARD_LOCKED {
                // Shard locked, spin
                std::thread::yield_now();
                continue;
            } else {
                let shard = unsafe { &*shard_ptr };
                return shard.get(slot, kh, key);
            }
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the key already exists, its value is replaced and the old value is returned.
    /// Otherwise, `None` is returned.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let (shard_index, slot, kh) = self.calc_index(&key);

        loop {
            let shard_ptr = self.shards[shard_index].load(Ordering::Relaxed);
            let shard_ptr_usize = shard_ptr as usize;

            if shard_ptr_usize != SHARD_LOCKED {
                let lock_result = self.shards[shard_index].compare_exchange_weak(
                    shard_ptr,
                    SHARD_LOCKED as *mut Shard<K, V>,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );

                if lock_result.is_ok() {
                    // Successfully locked
                    if shard_ptr_usize == SHARD_EMPTY {
                        let new_shard_ptr = Box::into_raw(Box::new(Shard::<K, V>::new()));
                        unsafe { &mut *new_shard_ptr }.insert(slot, kh, key, value);
                        self.shards[shard_index].store(new_shard_ptr, Ordering::Release);
                        return None;
                    } else {
                        let shard = unsafe { &mut *shard_ptr };
                        let result = shard.insert(slot, kh, key, value);
                        self.shards[shard_index].store(shard_ptr, Ordering::Release);
                        return result;
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
    use std::{hash::RandomState, sync::Arc, thread};

    use super::*;

    #[test]
    fn test_basic_operations() {
        let map: BFixMap<String, i32, RandomState> =
            BFixMap::<String, i32, RandomState>::with_capacity(10);

        // Insert
        assert_eq!(map.insert("one".to_string(), 1), None);
        assert_eq!(map.insert("two".to_string(), 2), None);

        // Get
        assert_eq!(map.get(&"one".to_string()), Some(1));
        assert_eq!(map.get(&"two".to_string()), Some(2));
        assert_eq!(map.get(&"three".to_string()), None);

        // // Modify
        // assert_eq!(map.modify(&"one".to_string(), |v| *v += 1), true);
        // assert_eq!(map.get(&"one".to_string()), Some(2));
        // assert_eq!(map.modify(&"three".to_string(), |v| *v += 1), false);

        // // // Remove
        // assert_eq!(map.remove(&"one".to_string()), Some(2));
        // assert_eq!(map.remove(&"two".to_string()), Some(2));
        // assert_eq!(map.get(&"one".to_string()), None);

        // Insert into deallocated
        assert_eq!(map.insert("three".to_string(), 1), None);
        assert_eq!(map.insert("four".to_string(), 2), None);

        assert_eq!(map.get(&"three".to_string()), Some(1));
        assert_eq!(map.get(&"four".to_string()), Some(2));
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
            assert_eq!(map.get(&i), Some(i));
        }
    }

    #[test]
    fn test_multithreaded_access() {
        const NUM_THREADS: usize = 100;
        const NUM_KEYS_PER_THREAD: usize = 10000;

        let map = Arc::new(
            BFixMap::<usize, usize, RandomState>::with_capacity_and_hasher(
                NUM_THREADS * NUM_KEYS_PER_THREAD,
                RandomState::default(),
            ),
        );

        let mut handles = vec![];

        for thread_id in 0..NUM_THREADS {
            let map_clone = Arc::clone(&map);
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
                assert_eq!(map.get(&key), Some(key));
            }
        }
    }
}
