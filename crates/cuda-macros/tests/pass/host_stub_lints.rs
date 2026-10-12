// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![deny(unused_imports, unused_macros, unused_mut, unused_attributes)]
#![allow(dead_code)]

#[cuda_macros::cuda_module]
mod kernels {
    use cuda_device::thread;
    macro_rules! write {
        ($out:ident) => {
            $out[thread::threadIdx_x() as usize] = 1.0;
        };
    }
    #[cuda_device::device]
    #[inline(always)]
    fn device_inline() -> u32 {
        1
    }
    #[cuda_device::kernel]
    pub fn direct(mut out: &mut [f32]) {
        let _ = device_inline();
        out = &mut out[1..];
        write!(out);
    }
    mod nested {
        use cuda_device::thread as device_thread;
        #[cuda_device::kernel]
        pub fn nested_entry(mut out: &mut [f32]) {
            out = &mut out[1..];
            out[device_thread::threadIdx_x() as usize] = 1.0;
        }
    }
}

fn main() {}
