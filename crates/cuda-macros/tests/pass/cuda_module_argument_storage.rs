/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exercise both sides of the shared argument-list grouping boundary.
use cuda_macros::{cuda_module, kernel};

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn empty() {}
    #[kernel]
    pub fn zero_sized(value: ()) {
        let _ = value;
    }
    #[kernel]
    pub fn args_32(
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
        a31: u32,
    ) {
        let _ = (
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18,
            a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
        );
    }
    #[kernel]
    pub fn args_33(
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
        a31: u32,
        a32: u32,
    ) {
        let _ = (
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18,
            a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32,
        );
    }
}

fn launch(module: &kernels::LoadedModule, stream: &cuda_core::CudaStream) {
    let config = cuda_core::simt::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.empty(stream, config).unwrap();
        module.zero_sized(stream, config, ()).unwrap();
        module
            .args_32(
                stream, config, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            )
            .unwrap();
        module
            .args_33(
                stream, config, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            )
            .unwrap();
    }
}

fn main() {
    let _ = launch;
}
