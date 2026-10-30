/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Selective convergence through device helper calls.
//!
//! Four paths exercise independent arithmetic, a shared-memory exchange, a
//! warp shuffle, and a loop-invariant thread-ID helper. The host checks exact
//! integer results. The companion verify-code-shape.sh checks both pre-opt
//! convergence attributes and the elimination of the invariant loop in PTX.
//!
//! input -> arithmetic_leaf -> arithmetic_wrapper -> output
//! shared store -> barrier_wrapper -> barrier_leaf -> shared neighbor load
//! lane value -> shuffle_wrapper -> shuffle_leaf -> neighboring lane value
//! repeated thread-ID helper -> one helper call and a multiply in optimized PTX
//!
//! Run with cargo oxide run device_convergence from the cuda-oxide workspace.

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

/// Each block has four full warps and one shared-memory slot per thread.
/// The fixed launch shape is part of the barrier and shuffle participation contract.
const BLOCK_SIZE: usize = 128;

/// Four blocks exercise both local lane indices and global output indices.
/// Every kernel writes exactly this many u32 values without a partial block.
const ELEMENTS: usize = BLOCK_SIZE * 4;

/// An arithmetic-only leaf with explicit modulo-2^32 behavior.
/// It has no inter-thread operations and needs no convergence restriction.
#[inline(never)]
pub fn arithmetic_leaf(value: u32) -> u32 {
    value.wrapping_mul(3).wrapping_add(7)
}

/// A second arithmetic-only call boundary, so classification must be transitive.
/// This wrapper stays independent even in a module containing collective helpers.
#[inline(never)]
pub fn arithmetic_wrapper(value: u32) -> u32 {
    arithmetic_leaf(value).wrapping_add(11)
}

/// Read this thread's fixed block-local ID and add a runtime base modulo 2^32.
/// A special-register read does not communicate with other threads or require convergence.
#[inline(never)]
pub fn thread_value_leaf(base: u32) -> u32 {
    thread::threadIdx_x().wrapping_add(base)
}

/// Keep a second call boundary above the thread-ID read.
/// Its value is invariant throughout a kernel, including inside the summation loop.
#[inline(never)]
pub fn thread_value_wrapper(base: u32) -> u32 {
    thread_value_leaf(base)
}

/// A block barrier reached by every launched thread at the same program point.
/// Its convergence requirement must propagate to callers independently of inlining.
#[inline(never)]
pub fn barrier_leaf() {
    thread::sync_threads();
}

/// An outer call boundary for the shared-memory exchange's block barrier.
/// All threads in each block must call this wrapper together, without early returns.
#[inline(never)]
pub fn barrier_wrapper() {
    barrier_leaf();
}

/// Exchange the value with lane XOR 1 using the full 32-lane member mask.
/// All lanes in every warp must participate together; the launch has no short warps.
#[inline(never)]
pub fn shuffle_leaf(value: u32) -> u32 {
    warp::shuffle_xor(value, 1)
}

/// Keep a second call boundary above the warp collective.
/// Callers inherit the leaf's full-warp participation and convergence requirements.
#[inline(never)]
pub fn shuffle_wrapper(value: u32) -> u32 {
    shuffle_leaf(value)
}

/// Four typed entries separate non-collective classification from collective paths.
/// Each uses the same complete 1D blocks and writes a disjoint global output slice.
#[cuda_module]
mod kernels {
    use super::*;

    /// Call arithmetic helpers from lane-dependent control flow.
    /// No barrier or shuffle is reachable, so this entry also needs no convergence.
    #[kernel]
    pub fn arithmetic_probe(mut output: DisjointSlice<u32>) {
        if let Some((slot, index)) = output.get_mut_indexed() {
            let i = index.get() as u32;
            *slot = if i & 1 == 0 {
                arithmetic_wrapper(i)
            } else {
                arithmetic_wrapper(i.wrapping_add(1))
            };
        }
    }

    /// Sum the same non-collective helper result a runtime number of times.
    /// Selective convergence allows LLVM to replace the loop with one call and a multiply.
    #[kernel]
    pub fn loop_invariant_probe(mut output: DisjointSlice<u32>, base: u32, iterations: u32) {
        let mut total = 0u32;
        let mut i = 0u32;
        while i < iterations {
            total = total.wrapping_add(thread_value_wrapper(base));
            i += 1;
        }
        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = total;
        }
    }

    /// Publish one value per shared slot, synchronize, then read the next slot.
    /// The host launches exactly BLOCK_SIZE threads per block; none skips the barrier.
    #[kernel]
    pub fn barrier_probe(mut output: DisjointSlice<u32>) {
        /// One u32 slot per thread, private to each CUDA block.
        /// Each slot has one writer before the barrier and a neighbor reader after it.
        static mut VALUES: SharedArray<u32, BLOCK_SIZE> = SharedArray::UNINIT;

        let local = thread::threadIdx_x() as usize;
        let index = thread::index_1d();
        // SAFETY: the launch has BLOCK_SIZE threads. Each writes its own shared
        // slot; all writes finish at the barrier before any neighboring read.
        unsafe { VALUES[local] = arithmetic_wrapper(index.get() as u32) };
        barrier_wrapper();
        let neighbor = (local + 1) % BLOCK_SIZE;
        // SAFETY: neighbor is in range and its slot was initialized before the barrier.
        let value = unsafe { VALUES[neighbor] };
        if let Some(slot) = output.get_mut(index) {
            *slot = value;
        }
    }

    /// Shuffle an arithmetic value through two helper boundaries.
    /// The collective executes before the output bounds check, with all lanes active.
    #[kernel]
    pub fn shuffle_probe(mut output: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let value = shuffle_wrapper(arithmetic_wrapper(index.get() as u32));
        if let Some(slot) = output.get_mut(index) {
            *slot = value;
        }
    }
}

/// Run all four paths and compare every result with an independent host oracle.
/// Integer operations and the small inputs make equality exact, with no timing claim.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let config = LaunchConfig {
        grid_dim: ((ELEMENTS / BLOCK_SIZE) as u32, 1, 1),
        block_dim: (BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut arithmetic = DeviceBuffer::<u32>::zeroed(&stream, ELEMENTS)?;
    let mut barrier = DeviceBuffer::<u32>::zeroed(&stream, ELEMENTS)?;
    let mut shuffle = DeviceBuffer::<u32>::zeroed(&stream, ELEMENTS)?;
    // SAFETY: all buffers cover the complete 1D launch. Every block has 128
    // threads and full warps; all threads reach the same barrier and shuffle.
    unsafe {
        module.arithmetic_probe(&stream, config, &mut arithmetic)?;
        module.barrier_probe(&stream, config, &mut barrier)?;
        module.shuffle_probe(&stream, config, &mut shuffle)?;
    }
    let results = [
        ("arithmetic", arithmetic.to_host_vec(&stream)?),
        ("barrier", barrier.to_host_vec(&stream)?),
        ("shuffle", shuffle.to_host_vec(&stream)?),
    ];
    for (path, (name, values)) in results.iter().enumerate() {
        for (i, &actual) in values.iter().enumerate() {
            let source = match path {
                0 => i + (i & 1),
                1 => i / BLOCK_SIZE * BLOCK_SIZE + (i + 1) % BLOCK_SIZE,
                _ => i ^ 1,
            };
            let expected = (source as u32).wrapping_mul(3).wrapping_add(18);
            assert_eq!(actual, expected, "{name}[{i}]");
        }
        println!("{name}: PASS ({ELEMENTS} exact results)");
    }

    let mut invariant = DeviceBuffer::<u32>::zeroed(&stream, ELEMENTS)?;
    let base = u32::MAX - 63;
    // End with zero iterations after nonzero results so an unwritten zero-case
    // output cannot pass merely because the buffer was initially zero-filled.
    for iterations in [1u32, 3, 17, 256, 0] {
        // SAFETY: the complete 1D launch and output length agree. This kernel
        // has no collective operations and accepts every u32 base and loop count.
        unsafe { module.loop_invariant_probe(&stream, config, &mut invariant, base, iterations)? };
        let values = invariant.to_host_vec(&stream)?;
        for (i, &actual) in values.iter().enumerate() {
            let expected = base
                .wrapping_add((i % BLOCK_SIZE) as u32)
                .wrapping_mul(iterations);
            assert_eq!(
                actual, expected,
                "loop_invariant[{i}], iterations={iterations}"
            );
        }
        println!("loop_invariant({iterations}): PASS ({ELEMENTS} exact results)");
    }
    println!("SUCCESS: all 4096 arithmetic, collective, and loop-invariant results match");
    Ok(())
}
