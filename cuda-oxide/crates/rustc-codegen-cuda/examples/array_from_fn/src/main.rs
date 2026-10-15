/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Minimal repro for device-prefix stripping inside Rust-mangled symbols.
//! The same capturing closure is specialized at lengths 1 and 4. Its marker
//! appears inside core::array::try_from_fn's mangled names; stripping up to
//! that marker used to erase the array length and collide during LLVM export.
//!
//! Run: cargo oxide run array_from_fn --arch sm_120

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    fn make_array<const N: usize>(base: u32) -> [u32; N] {
        core::array::from_fn(|i| base + i as u32)
    }

    #[kernel]
    pub fn array_from_fn(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let base = idx.get() as u32;
        if let Some(value) = out.get_mut(idx) {
            let one = make_array::<1>(base);
            let four = make_array::<4>(base);
            *value = one[0] + four[0] + four[1] + four[2] + four[3];
        }
    }
}

fn main() {
    const N: usize = 64;
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded CUDA module");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).expect("output buffer");
    // SAFETY: each lane accesses only its own element within the output buffer.
    unsafe {
        module.array_from_fn(
            stream.as_ref(),
            LaunchConfig::for_num_elems(N as u32),
            &mut out,
        )
    }
    .expect("array_from_fn launch");
    let got = out.to_host_vec(&stream).expect("copy output");
    for (lane, value) in got.into_iter().enumerate() {
        assert_eq!(value, 5 * lane as u32 + 6, "lane {lane}");
    }
    println!("array_from_fn: SUCCESS ({N} lanes, lengths 1 and 4)");
}
