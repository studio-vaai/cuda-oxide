/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime selection from thread-private arrays, including borrowed prepared data.
//! Run: `cargo oxide run register_array`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{RegisterArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    struct Prepared<const N: usize> {
        coefficients: RegisterArray<f32, N>,
    }

    impl<const N: usize> Prepared<N> {
        #[inline(always)]
        fn select(&self, role: usize) -> f32 {
            self.coefficients[role]
        }

        #[inline(always)]
        fn first(&self) -> f32 {
            self.coefficients[0]
        }

        #[inline(always)]
        fn last(&self) -> f32 {
            self.coefficients[N - 1]
        }
    }

    /// # Safety
    /// `input` must contain 3 floats; `indices` and `output` must contain
    /// one index and three output elements per launched thread. Every index must be less than 3.
    #[kernel]
    pub unsafe fn select_3(input: *const f32, indices: *const u32, output: *mut u32) {
        let i = thread::index_1d().get();
        unsafe {
            let prepared = Prepared {
                coefficients: RegisterArray::new([
                    input.add(0).read(),
                    input.add(1).read(),
                    input.add(2).read(),
                ]),
            };
            // The index is loaded at runtime, rather than bounded syntactically
            // by a remainder that the existing compiler scalarization recognizes.
            let role = indices.add(i).read() as usize;
            output.add(3 * i).write(prepared.select(role).to_bits());
            output.add(3 * i + 1).write(prepared.first().to_bits());
            output.add(3 * i + 2).write(prepared.last().to_bits());
        }
    }

    /// # Safety
    /// `input` must contain 4 floats; `indices` and `output` must contain
    /// one index and three output elements per launched thread. Every index must be less than 4.
    #[kernel]
    pub unsafe fn select_4(input: *const f32, indices: *const u32, output: *mut u32) {
        let i = thread::index_1d().get();
        unsafe {
            let prepared = Prepared {
                coefficients: RegisterArray::new([
                    input.add(0).read(),
                    input.add(1).read(),
                    input.add(2).read(),
                    input.add(3).read(),
                ]),
            };
            // The index is loaded at runtime, rather than bounded syntactically
            // by a remainder that the existing compiler scalarization recognizes.
            let role = indices.add(i).read() as usize;
            output.add(3 * i).write(prepared.select(role).to_bits());
            output.add(3 * i + 1).write(prepared.first().to_bits());
            output.add(3 * i + 2).write(prepared.last().to_bits());
        }
    }

    /// # Safety
    /// `input` must contain 16 floats; `indices` and `output` must contain
    /// one index and three output elements per launched thread. Every index must be less than 16.
    #[kernel]
    pub unsafe fn select_16(input: *const f32, indices: *const u32, output: *mut u32) {
        let i = thread::index_1d().get();
        unsafe {
            let prepared = Prepared {
                coefficients: RegisterArray::new([
                    input.add(0).read(),
                    input.add(1).read(),
                    input.add(2).read(),
                    input.add(3).read(),
                    input.add(4).read(),
                    input.add(5).read(),
                    input.add(6).read(),
                    input.add(7).read(),
                    input.add(8).read(),
                    input.add(9).read(),
                    input.add(10).read(),
                    input.add(11).read(),
                    input.add(12).read(),
                    input.add(13).read(),
                    input.add(14).read(),
                    input.add(15).read(),
                ]),
            };
            // The index is loaded at runtime, rather than bounded syntactically
            // by a remainder that the existing compiler scalarization recognizes.
            let role = indices.add(i).read() as usize;
            output.add(3 * i).write(prepared.select(role).to_bits());
            output.add(3 * i + 1).write(prepared.first().to_bits());
            output.add(3 * i + 2).write(prepared.last().to_bits());
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");
    let bits = [
        0_u32, 0x80000000, 0x7fc00123, 0xff800000, 1, 0x3f800000, 0xbf800000, 0x7f800000,
        0x00800000, 0x7f7fffff, 0xff7fffff, 0x3f000000, 0x40000000, 0x40400000, 0x40800000,
        0x40a00000,
    ];
    let values = bits.map(f32::from_bits);
    let input = DeviceBuffer::from_host(&stream, &values).expect("upload values");
    for n in [3_u32, 4, 16] {
        let indices: Vec<u32> = (0..256).map(|i| (i * 7 + 1) % n).collect();
        let roles = DeviceBuffer::from_host(&stream, &indices).expect("upload indices");
        let output = DeviceBuffer::<u32>::zeroed(&stream, 3 * indices.len()).expect("alloc output");
        let config = LaunchConfig::for_num_elems(indices.len() as u32);
        // SAFETY: input has 16 floats, roles has 256 elements, output has 768 elements, and all
        // uploaded indices are valid for the selected kernel's array length.
        unsafe {
            let pointers = (
                input.cu_deviceptr() as *const f32,
                roles.cu_deviceptr() as *const u32,
                output.cu_deviceptr() as *mut u32,
            );
            match n {
                3 => module.select_3(&stream, config, pointers.0, pointers.1, pointers.2),
                4 => module.select_4(&stream, config, pointers.0, pointers.1, pointers.2),
                16 => module.select_16(&stream, config, pointers.0, pointers.1, pointers.2),
                _ => unreachable!(),
            }
            .expect("launch");
        }
        stream.synchronize().expect("sync");
        let actual = output.to_host_vec(&stream).expect("download output");
        let expected: Vec<u32> = indices
            .iter()
            .flat_map(|&i| [bits[i as usize], bits[0], bits[n as usize - 1]])
            .collect();
        assert_eq!(actual, expected, "RegisterArray length {n}");
    }
    println!("register_array: PASS (3/4/16 elements, runtime/constant indices, exact float bits)");
}
