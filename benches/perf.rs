use bfixmap::BFixMap;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rand::Rng; 

fn bench_insert(c: &mut Criterion) {
    let map = BFixMap::<usize, usize>::with_capacity(1000); // Adjust as needed

    c.bench_function("insert", |b| {
        b.iter(|| {
            map.insert(black_box(1), black_box(1)); // Benchmark insertion
        })
    });
}

fn bench_read_95(c: &mut Criterion) {
    const NUM_KEYS: usize = 100_000;
    const NUM_ITERATIONS: usize = 1000;

    let map = BFixMap::<usize, usize>::with_capacity(NUM_KEYS);

    // Pre-populate the map with some data
    for i in 0..NUM_KEYS {
        map.insert(i, i);
    }

    let mut rng = rand::thread_rng();

    c.bench_function("mixed_operations_5_95", |b| {
        b.iter(|| {
            for _ in 0..NUM_ITERATIONS {
                let key = rng.gen_range(0..NUM_KEYS);
                if rng.gen_range(0..100) < 5 { // 5% chance of insert
                    map.insert(black_box(key), black_box(key)); 
                } else { // 95% chance of read
                    map.get(&black_box(key), |v| v.clone());
                }
            }
        })
    });
}

fn bench_read_100_large(c: &mut Criterion) {
    const NUM_KEYS: usize = 1_000_000; 
    const NUM_READS: usize = 1000;

    let map = BFixMap::<usize, usize, ahash::RandomState>::with_capacity(NUM_KEYS);

    for i in 0..NUM_KEYS {
        map.insert(i, i);
    }

    let mut rng = rand::thread_rng();

    c.bench_function("read_100_large_collection", |b| {
        b.iter(|| {
            for _ in 0..NUM_READS {
                let key = rng.gen_range(0..NUM_KEYS);
                map.get(&black_box(key), |v| v.clone());
            }
        })
    });
}

fn bench_read_100_small(c: &mut Criterion) {
    const NUM_KEYS: usize = 10_000; 
    const NUM_READS: usize = 1000;

    let map = BFixMap::<usize, usize, ahash::RandomState>::with_capacity(NUM_KEYS);

    for i in 0..NUM_KEYS {
        map.insert(i, i);
    }

    let mut rng = rand::thread_rng();

    c.bench_function("read_100_small_collection", |b| {
        b.iter(|| {
            for _ in 0..NUM_READS {
                let key = rng.gen_range(0..NUM_KEYS);
                map.get(&black_box(key), |v| v.clone());
            }
        })
    });
}

criterion_group!(benches, bench_insert, bench_read_95, bench_read_100_large, bench_read_100_small);
criterion_main!(benches);