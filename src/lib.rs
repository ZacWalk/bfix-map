use std::hash::{BuildHasher, Hash, Hasher, RandomState};
use std::mem;

type BFixLock<T> = parking_lot::Mutex<T>;

struct Entry<K, V> {
    key: K,
    value: Option<V>,
}

pub struct BFixMap<K, V: Clone, S = RandomState> {
    buckets: Vec<BFixLock<Vec<Entry<K, V>>>>,
    build_hasher: S,
    bucket_count: usize,
}

fn closest_power_of_2_min_1024(value: usize) -> usize {
    // Ensure the minimum value is 1024
    let value = value.max(1024);

    if value.is_power_of_two() {
        return value;
    }

    // Find the next power of 2
    let next_power_of_2 = 1 << (usize::BITS - value.leading_zeros());

    // Calculate the previous power of 2
    let prev_power_of_2 = next_power_of_2 >> 1;

    // Determine which power of 2 is closer
    if (value - prev_power_of_2) <= (next_power_of_2 - value) {
        prev_power_of_2
    } else {
        next_power_of_2
    }
}

/// A concurrent hash map with bucket-level fine-grained locking.
///
/// This map is optimized to provide safe concurrent access for multiple threads, allowing
/// simultaneous reads and writes without blocking the entire map.
/// 
/// This map has a naive implementation however it turns out to have very good performance 
/// with large numbers of threads. The trade-off is that the number of buckets is set at 
/// creation time based on the provided capacity. The collection can grow to contain larger 
/// numbers of items than the specified capacity, but the number of buckets does not change. 
/// This design avoids any complex mechanisms around splitting buckets, reducing lock contention.
///
/// # Type Parameters
///
/// * `K`: The type of keys stored in the map. Must implement `Hash` and `Eq`.
/// * `V`: The type of values stored in the map. Must implement `Clone`.
/// * `S`: The type of build hasher used for hashing keys. Defaults to `RandomState`.
impl<K: Hash + Eq, V: Clone, S: BuildHasher + Default> BFixMap<K, V, S> {

    /// Creates a new `BFixMap` with the specified capacity and build hasher.
    pub fn with_capacity_and_hasher(capacity: usize, build_hasher: S) -> Self {
        let bucket_count = closest_power_of_2_min_1024(capacity / 256);
        let mut buckets = Vec::with_capacity(bucket_count);
        buckets.resize_with(bucket_count, || BFixLock::new(Vec::new()));

        Self {
            buckets,
            build_hasher,
            bucket_count,
        }
    }

    /// Creates a new `BFixMap` with the specified capacity and a default build hasher.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }

    fn hash_key(&self, key: &K) -> usize {
        let mut hasher = self.build_hasher.build_hasher();
        key.hash(&mut hasher);
        let mask = self.bucket_count - 1;
        hasher.finish() as usize & mask
    }

    /// Retrieves the value associated with the given key, if it exists.
    pub fn get(&self, key: &K) -> Option<V> {
        let hash_key = self.hash_key(&key);
        let buckets = &self.buckets;
        let bucket = &buckets[hash_key].lock();
        bucket
            .iter()
            .find(|entry| entry.key == *key)
            .map(|entry| entry.value.clone())?
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the key already exists, its value is replaced and the old value is returned.
    /// Otherwise, `None` is returned.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let hash_key = self.hash_key(&key);
        let buckets = &self.buckets;
        let bucket = &mut buckets[hash_key].lock();
        for entry in bucket.iter_mut() {
            if entry.key == key {
                return mem::replace(&mut entry.value, Some(value));
            }
        }
        bucket.push(Entry {
            key,
            value: Some(value),
        });
        None
    }

    /// Removes the key-value pair associated with the given key, if it exists.
    ///
    /// If the key exists, its value is removed and returned. Otherwise, `None` is returned.
    pub fn remove(&self, key: &K) -> Option<V> {
    pub fn remove(&self, key: &K) -> Option<V> {
        let hash_key = self.hash_key(&key);
        let buckets = &self.buckets;
        let bucket = &mut buckets[hash_key].lock();
        for i in 0..bucket.len() {
            if bucket[i].key == *key {
                return mem::replace(&mut bucket[i].value, None);
            }
        }
        None
    }

    /// Modifies the value associated with the given key using the provided function.
    ///
    /// If the key exists, the function `f` is called with a mutable reference to the value.
    /// Returns `true` if the value was modified, `false` otherwise.
    pub fn modify<F>(&self, key: &K, f: F) -> bool
    where
        F: FnOnce(&mut V),
    {
        let hash_key = self.hash_key(&key);
        let buckets = &self.buckets;
        let bucket = &mut buckets[hash_key].lock();
        let mut modified = false;

        bucket
            .iter_mut()
            .find(|entry| entry.key == *key)
            .map(|entry| {
                if let Some(ref mut value) = entry.value {
                    // Check if value exists
                    f(value);
                    modified = true;
                }
            });

        modified
    }
}

#[cfg(test)]
mod tests {
    use std::{hash::RandomState, sync::Arc, thread};

    use super::*;

    #[test]
    fn test_basic_operations() {
        let map: BFixMap<String, i32, RandomState> = BFixMap::<String, i32, RandomState>::with_capacity(10);

        // Insert
        assert_eq!(map.insert("one".to_string(), 1), None);
        assert_eq!(map.insert("two".to_string(), 2), None);

        // Get
        assert_eq!(map.get(&"one".to_string()), Some(1));
        assert_eq!(map.get(&"two".to_string()), Some(2));
        assert_eq!(map.get(&"three".to_string()), None);

        // Modify
        assert_eq!(map.modify(&"one".to_string(), |v| *v += 1), true);
        assert_eq!(map.get(&"one".to_string()), Some(2));
        assert_eq!(map.modify(&"three".to_string(), |v| *v += 1), false);

        // Remove
        assert_eq!(map.remove(&"one".to_string()), Some(2));
        assert_eq!(map.get(&"one".to_string()), None);
    }

    #[test]
    fn test_large_capacity() {
        const CAPACITY: usize = 1_000_000;
        let map = BFixMap::<usize, usize, RandomState>::with_capacity_and_hasher(CAPACITY, RandomState::new());

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
        const NUM_KEYS_PER_THREAD: usize = 1000;

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
