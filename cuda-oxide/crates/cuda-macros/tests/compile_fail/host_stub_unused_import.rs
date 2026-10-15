// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![deny(unused_imports)]
#![allow(dead_code)]

#[cuda_macros::cuda_module]
mod kernels {
    use cuda_device::{SharedArray, thread};
    #[cuda_device::kernel]
    pub fn direct(out: &mut [f32]) {
        out[thread::threadIdx_x() as usize] = 1.0;
    }
}

fn main() {}
