/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CuSimd<T, N> vector load/store lowering probe.
//!
//! Goal: check whether loading/storing a whole `CuSimd<T, N>` through global
//! memory lowers to the vectorized PTX data-movement instructions
//! (`ld.global.v2`, `ld.global.v4`, `st.global.v2`, `st.global.v4`) or whether
//! it falls back to a sequence of scalar `ld.global` / `st.global`.
//!
//! See the PTX ISA, "Data Movement and Conversion Instructions: ld" and the
//! `.vec` (`.v2`/`.v4`) qualifier:
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld>
//!
//! Each kernel has the same shape: read one `CuSimd<T, N>` element from an
//! input slice and write it back to an output slice at the thread's index.
//! The load of `input[i]` is a whole-vector global load; the store `*o = v`
//! is a whole-vector global store. PTX vectorization of these accesses depends
//! on the access width *and the alignment of the type*: `CuSimd<T, N>` is
//! `#[repr(C)] { data: [T; N] }`, so its alignment is `align_of::<T>()`, not
//! the vector width. We measure the result as-is (no alignment changes).
//!
//! Run: `cargo oxide build cusimd_ops` then inspect `cusimd_ops.ptx`.
//!
//! Combination matrix probed below:
//!   - 32-bit elems (f32, u32, i32): N = 2 (v2/64-bit), N = 4 (v4/128-bit)
//!   - 64-bit elems (f64, u64, i64): N = 2 (v2/128-bit), N = 4 (256-bit -> 2x v2)
//!   - 16-bit elems (u16, i16):      N = 2 (v2/32-bit),  N = 4 (v4/64-bit)
//!   - large-N f32:                  N = 8, 16, 32 (must split into multiple v4)

use cuda_device::cusimd::CuSimd;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    // Each kernel below is identical in shape:
    //
    //   let idx = thread::index_1d();
    //   let i = idx.get();
    //   if let Some(o) = output.get_mut(idx) {
    //       *o = input[i];   // whole-vector global load, then global store
    //   }
    //
    // Only the element type T and lane count N vary. `input[i]` is the
    // whole-vector load we want coalesced into `ld.global.v{2,4}`; `*o = ...`
    // is the store we want coalesced into `st.global.v{2,4}`.

    // ===== 32-bit element types: f32 / u32 / i32 =====

    #[kernel]
    pub fn ls_f32x2(input: &[CuSimd<f32, 2>], mut output: DisjointSlice<CuSimd<f32, 2>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_f32x4(input: &[CuSimd<f32, 4>], mut output: DisjointSlice<CuSimd<f32, 4>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_u32x2(input: &[CuSimd<u32, 2>], mut output: DisjointSlice<CuSimd<u32, 2>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_u32x4(input: &[CuSimd<u32, 4>], mut output: DisjointSlice<CuSimd<u32, 4>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_i32x2(input: &[CuSimd<i32, 2>], mut output: DisjointSlice<CuSimd<i32, 2>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_i32x4(input: &[CuSimd<i32, 4>], mut output: DisjointSlice<CuSimd<i32, 4>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    // ===== 64-bit element types: f64 / u64 / i64 =====

    #[kernel]
    pub fn ls_f64x2(input: &[CuSimd<f64, 2>], mut output: DisjointSlice<CuSimd<f64, 2>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_f64x4(input: &[CuSimd<f64, 4>], mut output: DisjointSlice<CuSimd<f64, 4>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_u64x2(input: &[CuSimd<u64, 2>], mut output: DisjointSlice<CuSimd<u64, 2>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_u64x4(input: &[CuSimd<u64, 4>], mut output: DisjointSlice<CuSimd<u64, 4>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_i64x2(input: &[CuSimd<i64, 2>], mut output: DisjointSlice<CuSimd<i64, 2>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_i64x4(input: &[CuSimd<i64, 4>], mut output: DisjointSlice<CuSimd<i64, 4>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    // ===== 16-bit element types: u16 / i16 =====

    #[kernel]
    pub fn ls_u16x2(input: &[CuSimd<u16, 2>], mut output: DisjointSlice<CuSimd<u16, 2>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_u16x4(input: &[CuSimd<u16, 4>], mut output: DisjointSlice<CuSimd<u16, 4>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_i16x2(input: &[CuSimd<i16, 2>], mut output: DisjointSlice<CuSimd<i16, 2>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_i16x4(input: &[CuSimd<i16, 4>], mut output: DisjointSlice<CuSimd<i16, 4>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    // ===== large-N f32 (must split into multiple v4 loads/stores) =====

    #[kernel]
    pub fn ls_f32x8(input: &[CuSimd<f32, 8>], mut output: DisjointSlice<CuSimd<f32, 8>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_f32x16(input: &[CuSimd<f32, 16>], mut output: DisjointSlice<CuSimd<f32, 16>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn ls_f32x32(input: &[CuSimd<f32, 32>], mut output: DisjointSlice<CuSimd<f32, 32>>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================
//
// No GPU launch is required to inspect the generated PTX: `cargo oxide build`
// emits `cusimd_ops.ptx` containing every kernel above. `main` exists only so
// this is a buildable binary crate; the #[cuda_module] glue compiles against
// the host CUDA runtime but is not exercised here.
fn main() {
    println!("cusimd_ops: device kernels compiled.");
    println!("Inspect cusimd_ops.ptx for ld.global.v2/v4 and st.global.v2/v4.");
}
