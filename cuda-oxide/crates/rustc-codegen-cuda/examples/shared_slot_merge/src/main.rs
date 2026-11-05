/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared and global pointers merged into one slot
//!
//! `field` holds a dynamic shared-memory pointer on one path and a pointer
//! derived from a slice parameter on the other. The slice path reaches the
//! slot through pointer casts whose sources never carry a classified address
//! space, so the slot must stay generic. A closure then captures `field` by
//! reference and an out-of-line helper reads through it: if the slot were
//! narrowed to `addrspace(3)`, the captured reference would not match the
//! closure's generic field and device MIR verification would fail.
//!
//! Build and run with:
//!   cargo oxide run shared_slot_merge

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Sums `read(0..len)`; out of line so the closure stays a real capture.
    #[inline(never)]
    fn sum_window(len: usize, read: impl Fn(usize) -> u32) -> u32 {
        let mut total = 0;
        let mut i = 0;
        while i < len {
            total += read(i);
            i += 1;
        }
        total
    }

    /// Every thread writes the sum of the block's window, read from doubled
    /// shared copies when `use_shared != 0` and from `input` otherwise.
    #[kernel]
    pub fn shared_slot_merge(input: &[u32], mut out: DisjointSlice<u32>, use_shared: u32) {
        let tid = thread::threadIdx_x() as usize;
        let len = thread::blockDim_x() as usize;
        let smem: *mut u32 = DynamicSharedArray::<u32>::get();
        unsafe { *smem.add(tid) = 2 * input[tid] };
        thread::sync_threads();
        let field: *const u32 = if use_shared != 0 {
            smem
        } else {
            input.as_ptr()
        };
        let total = sum_window(len, |i| unsafe { *field.add(i) });
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = total;
        }
    }
}

fn main() {
    const N: usize = 128;
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let input: Vec<u32> = (1..=N as u32).collect();
    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let sum: u32 = input.iter().sum();
    for (use_shared, expected) in [(0u32, sum), (1u32, 2 * sum)] {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: (N * core::mem::size_of::<u32>()) as u32,
        };
        // SAFETY: one block of N threads; `input` and `out` hold N elements
        // and the dynamic shared allocation holds N words.
        unsafe { module.shared_slot_merge(stream.as_ref(), cfg, &input_dev, &mut out_dev, use_shared) }
            .expect("Kernel launch failed");
        let out = out_dev.to_host_vec(&stream).unwrap();
        if let Some((i, got)) = out.iter().enumerate().find(|(_, got)| **got != expected) {
            eprintln!("use_shared={use_shared}: out[{i}] = {got}, expected {expected}");
            std::process::exit(1);
        }
        println!("use_shared={use_shared}: every thread read {expected}");
    }
    println!("SUCCESS: shared and slice pointers merge through a generic slot");
}
