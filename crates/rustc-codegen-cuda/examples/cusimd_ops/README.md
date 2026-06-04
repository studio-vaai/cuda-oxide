# cusimd_ops — does `CuSimd<T, N>` lower to vector PTX?

A lowering probe: each kernel loads one whole `CuSimd<T, N>` from a global
`&[CuSimd<T, N>]` and stores it back through a `DisjointSlice<CuSimd<T, N>>`
(`output[idx] = input[idx]`). The intent is to inspect the generated PTX and
check whether these whole-vector accesses lower to the vectorized
data-movement instructions described in the PTX ISA —
[`ld` + the `.vec` (`.v2`/`.v4`) qualifier][ld] — or fall back to scalar loads.

[ld]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld

## Build & inspect

```bash
cargo oxide build cusimd_ops            # emits cusimd_ops.ptx (no GPU needed)
./analyze.sh                            # per-kernel ld/st widths + vector check
# or by hand:
grep -nE 'ld\.global|st\.global|\.v2|\.v4' cusimd_ops.ptx
```

## Combination matrix

All eight `SimdElement` types × `N ∈ {2, 4}`, plus large-N `f32`:

| Element (size) | N=2 | N=4 | large N |
|---|---|---|---|
| `f32` / `u32` / `i32` (32-bit) | v2 = 64-bit | v4 = 128-bit | `f32`×8/16/32 |
| `f64` / `u64` / `i64` (64-bit) | v2 = 128-bit | 256-bit → 2× v2 | — |
| `u16` / `i16` (16-bit) | v2 = 32-bit | v4 = 64-bit | — |

(19 kernels total.)

## Findings (measured on sm_80 via NVPTX back-end, LLVM 18)

**`CuSimd<T, N>` does NOT currently lower to vectorized PTX.** Across all 19
kernels there is **zero** `.global`, `.v2`, `.v4`, or `.nc` in the output. The
whole-`CuSimd` assignment `*o = input[i]` lowers to a **generic-state-space,
byte-blob `memcpy`** — scalar `ld.b64`/`st.b64` chunks (and a literal
byte-copy *loop* for the largest type):

| Kernel | bytes | data loads/stores emitted |
|---|---|---|
| `ls_u16x2` / `ls_i16x2` | 4 | 1× `ld.b32` + 1× `st.b32` |
| `ls_f32x2` / `ls_u32x2` / `ls_i32x2` | 8 | 1× `ld.b64` + 1× `st.b64` |
| `ls_u16x4` / `ls_i16x4` | 8 | 1× `ld.b64` + 1× `st.b64` |
| `ls_f32x4` / `ls_u32x4` / `ls_i32x4` | 16 | 2× `ld.b64` + 2× `st.b64` |
| `ls_f64x2` / `ls_u64x2` / `ls_i64x2` | 16 | 2× `ld.b64` + 2× `st.b64` |
| `ls_f32x8` | 32 | 4× `ld.b64` + 4× `st.b64` |
| `ls_f64x4` / `ls_u64x4` / `ls_i64x4` | 32 | 4× `ld.b64` + 4× `st.b64` |
| `ls_f32x16` | 64 | 8× `ld.b64` + 8× `st.b64` |
| `ls_f32x32` | 128 | **byte-copy loop**: `ld.b8`/`st.b8` × 128 |

Example — `ls_f32x4` (the “float4” case, where one would hope for
`ld.global.v4.f32`):

```ptx
.visible .entry ls_f32x4(
    .param .u64 .ptr .align 1 ls_f32x4_param_0,   // <- pointer passed as align 1
    ...
)
    ...
    ld.b64  %rd17, [%rd16+8];    // generic space, scalar, unaligned
    ld.b64  %rd18, [%rd16];
    st.b64  [%rd19],   %rd18;
    st.b64  [%rd19+8], %rd17;
```

### Why it doesn't vectorize (three independent blockers, all visible in PTX)

1. **Struct copy → `memcpy`.** `*o = input[i]` is a whole-aggregate move, which
   LLVM lowers to `llvm.memcpy`. NVPTX expands that to generic word/byte copies,
   not a typed vector load/store. (The element type is never reconstructed as a
   vector.)
2. **Pointers are `.align 1`.** Kernel params are emitted as `.ptr .align 1`,
   so the back-end may not assume any alignment. `ld.global.v4.f32` requires
   16-byte alignment; even `v2` needs 8. With align-1, vectorization is illegal.
   (`CuSimd` is `#[repr(C)] { data: [T; N] }`, so its *Rust* alignment is only
   `align_of::<T>()` anyway — 4 for `f32`, never the 16 a `float4` wants.)
3. **No `.global` state space.** Loads are generic (`ld.b64`, not
   `ld.global.b64`), so the read-only / non-coherent path (`ld.global.nc`,
   i.e. `__ldg`) is off the table regardless.

### Implications / possible follow-ups (not done here — we measured as-is)

- Give `CuSimd<T, N>` a vector-width alignment (e.g. `#[repr(C, align(16))]`
  for the 128-bit configs) **and** propagate that alignment to the kernel-ABI
  pointer params, so the access is at least `.align 16`.
- Lower whole-`CuSimd` loads/stores as typed vector ops instead of `memcpy`
  (so the back-end sees a `<4 x float>` load), and tag the address space as
  `.global`.
- A read-only (`ld.global.nc` / `__ldg`) path needs invariant-load / readonly
  noalias metadata, which does not exist in the pipeline today — separate task.

Reproduce the table any time with `./analyze.sh`.
