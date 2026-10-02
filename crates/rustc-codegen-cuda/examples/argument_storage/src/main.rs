/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Check scalar/ZST/slice ABI order across the 32-field argument-group boundary.
//! Run: cargo oxide run argument_storage
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use cuda_device::kernel;
    // This ABI fixture deliberately crosses the 32-field grouping boundary.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn readback(
        a0: u32,
        a1: u32,
        a2: u32,
        a3: u32,
        a4: u32,
        a5: u32,
        a6: u32,
        a7: u32,
        a8: u32,
        a9: u32,
        a10: u32,
        a11: u32,
        a12: u32,
        a13: u32,
        a14: u32,
        a15: u32,
        a16: u32,
        a17: u32,
        a18: u32,
        a19: u32,
        a20: u32,
        a21: u32,
        a22: u32,
        a23: u32,
        a24: u32,
        a25: u32,
        a26: u32,
        a27: u32,
        a28: u32,
        a29: u32,
        a30: u32,
        _zero: (),
        out: &mut [u32],
        tail: u64,
    ) {
        let values = [
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18,
            a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30,
        ];
        out[..31].copy_from_slice(&values);
        out[31] = tail as u32;
        out[32] = (tail >> 32) as u32;
        out[33] = out.len() as u32;
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).unwrap();
    let mut out = DeviceBuffer::from_host(&stream, &[0u32; 34]).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: a single thread writes only the 34 allocated output elements.
    unsafe {
        module
            .readback(
                &stream,
                config,
                100,
                103,
                106,
                109,
                112,
                115,
                118,
                121,
                124,
                127,
                130,
                133,
                136,
                139,
                142,
                145,
                148,
                151,
                154,
                157,
                160,
                163,
                166,
                169,
                172,
                175,
                178,
                181,
                184,
                187,
                190,
                (),
                &mut out,
                0xaabb_ccdd_1234_5678,
            )
            .unwrap();
    }
    let actual = out.to_host_vec(&stream).unwrap();
    let mut expected: Vec<u32> = (0..31).map(|i| 100 + i * 3).collect();
    expected.extend([0x1234_5678, 0xaabb_ccdd, 34]);
    assert_eq!(actual, expected);
    println!("PASS: 34 logical arguments, skipped ZST, slice length and 64-bit tail across groups");
}
