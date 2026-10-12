/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Value selection for small thread-private arrays.

/// A small fixed-size array whose runtime reads use constant element accesses.
///
/// Unlike ordinary runtime array indexing, [`get`](Self::get) selects between
/// literal element reads. This lets the device optimizer keep a thread-private
/// array as independent register values instead of making it addressable in
/// local memory. Actual register allocation still depends on the surrounding
/// kernel and register pressure.
///
/// Supports lengths from zero through 16. Elements must be [`Copy`]. `get`
/// returns a value; `Index` returns an element reference using literal indices.
/// Immediate indexing reads can be promoted just like value selection, while
/// retaining or escaping references can require addressable storage. There is
/// no `AsRef` or `Deref` implementation. Use ordinary arrays or memory views
/// for global/shared memory and larger arrays.
///
/// ```
/// use cuda_device::RegisterArray;
///
/// let coefficients = RegisterArray::new([1.0_f32, 2.0, 3.0]);
/// let role = 1;
/// assert_eq!(coefficients.get(role), 2.0);
/// assert_eq!(coefficients[role], 2.0);
/// assert_eq!(coefficients[2], 3.0);
/// assert_eq!(coefficients.get_const::<2>(), 3.0);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterArray<T: Copy, const N: usize> {
    values: [T; N],
}

impl<T: Copy, const N: usize> RegisterArray<T, N> {
    /// Construct from values. Lengths greater than 16 are rejected at compile time.
    ///
    /// ```compile_fail
    /// use cuda_device::RegisterArray;
    /// const TOO_LARGE: RegisterArray<u32, 17> = RegisterArray::new([0; 17]);
    /// ```
    #[inline(always)]
    pub const fn new(values: [T; N]) -> Self {
        const { assert!(N <= 16, "RegisterArray supports at most 16 elements") };
        Self { values }
    }

    /// Select an element by value using constant array indices.
    ///
    /// # Panics
    /// Panics if `index >= N`.
    #[inline(always)]
    pub fn get(self, index: usize) -> T {
        self[index]
    }

    /// Read a compile-time element index by value.
    ///
    /// An out-of-bounds index is rejected at compile time.
    ///
    /// ```compile_fail
    /// use cuda_device::RegisterArray;
    /// const INVALID: u32 = RegisterArray::new([10, 20, 30]).get_const::<3>();
    /// ```
    #[inline(always)]
    pub const fn get_const<const I: usize>(self) -> T {
        const { assert!(I < N, "RegisterArray index out of bounds") };
        self.values[I]
    }

    /// Return the underlying array by value.
    #[inline(always)]
    pub const fn into_array(self) -> [T; N] {
        self.values
    }
}

impl<T: Copy, const N: usize> core::ops::Index<usize> for RegisterArray<T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &T {
        match index {
            0 if N > 0 => &self.values[0],
            1 if N > 1 => &self.values[1],
            2 if N > 2 => &self.values[2],
            3 if N > 3 => &self.values[3],
            4 if N > 4 => &self.values[4],
            5 if N > 5 => &self.values[5],
            6 if N > 6 => &self.values[6],
            7 if N > 7 => &self.values[7],
            8 if N > 8 => &self.values[8],
            9 if N > 9 => &self.values[9],
            10 if N > 10 => &self.values[10],
            11 if N > 11 => &self.values[11],
            12 if N > 12 => &self.values[12],
            13 if N > 13 => &self.values[13],
            14 if N > 14 => &self.values[14],
            15 if N > 15 => &self.values[15],
            _ => panic!("RegisterArray index out of bounds"),
        }
    }
}

impl<T: Copy, const N: usize> From<[T; N]> for RegisterArray<T, N> {
    #[inline(always)]
    fn from(values: [T; N]) -> Self {
        Self::new(values)
    }
}

#[cfg(test)]
mod tests {
    use super::RegisterArray;

    #[test]
    fn selection_preserves_float_bits() {
        let values = [0.0_f32, -0.0, f32::from_bits(0x7fc00123), f32::NEG_INFINITY];
        let registers = RegisterArray::new(values);
        for (index, value) in values.into_iter().enumerate() {
            assert_eq!(registers.get(index).to_bits(), value.to_bits());
            assert_eq!(registers[index].to_bits(), value.to_bits());
        }
        let triangle = RegisterArray::new([values[0], values[1], values[2]]);
        for index in 0..3 {
            assert_eq!(
                triangle.get(index).to_bits(),
                registers.get(index).to_bits()
            );
        }
    }

    #[test]
    fn supports_all_lengths_and_copy_elements() {
        fn check<const N: usize>() {
            let values: [u32; N] = core::array::from_fn(|i| i as u32 + 10);
            let registers = RegisterArray::from(values);
            for (index, value) in values.into_iter().enumerate() {
                assert_eq!(registers.get(index), value);
                assert_eq!(registers[index], value);
            }
            assert_eq!(registers.into_array(), values);
        }
        check::<0>();
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<5>();
        check::<6>();
        check::<7>();
        check::<8>();
        check::<9>();
        check::<10>();
        check::<11>();
        check::<12>();
        check::<13>();
        check::<14>();
        check::<15>();
        check::<16>();
        assert_eq!(RegisterArray::new([(1_u32, 2_u32)])[0], (1, 2));
        assert_eq!(core::mem::size_of::<RegisterArray<f32, 3>>(), 12);
        assert_eq!(core::mem::align_of::<RegisterArray<f32, 3>>(), 4);
    }

    #[test]
    fn constant_access() {
        const REGISTERS: RegisterArray<u32, 3> = RegisterArray::new([10, 20, 30]);
        const LAST: u32 = REGISTERS.get_const::<2>();
        assert_eq!(LAST, 30);
        assert_eq!(REGISTERS[2], LAST);
    }

    #[test]
    #[should_panic(expected = "RegisterArray index out of bounds")]
    fn empty_rejects_read() {
        RegisterArray::<u32, 0>::new([]).get(0);
    }

    #[test]
    #[should_panic(expected = "RegisterArray index out of bounds")]
    fn triangle_rejects_fourth_role() {
        RegisterArray::new([1.0, 2.0, 3.0]).get(3);
    }

    #[test]
    #[should_panic(expected = "RegisterArray index out of bounds")]
    fn rejects_large_index() {
        RegisterArray::new([1.0, 2.0, 3.0, 4.0]).get(usize::MAX);
    }
    #[test]
    #[should_panic(expected = "RegisterArray index out of bounds")]
    fn index_rejects_out_of_bounds() {
        let _ = RegisterArray::new([1.0, 2.0, 3.0])[3];
    }

    #[test]
    #[should_panic(expected = "RegisterArray index out of bounds")]
    fn index_rejects_empty() {
        let _ = RegisterArray::<u32, 0>::new([])[0];
    }
}
