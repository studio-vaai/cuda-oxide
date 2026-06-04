# simd_vec_ops — vector load/store lowering test

Verifies that whole-vector loads/stores lower to the **vectorized** PTX
data-movement instructions ([`ld`/`st` + the `.vec` (`.v2`/`.v4`) qualifier][ld])
— **soundly**.

[ld]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld

Each `v_*` kernel does `output[idx] = input[idx]` over a genuine
`#[repr(simd)]` vector type `<N x T>`. The codegen lowers Vector-ABI types to a
real LLVM vector (see the Vector-ABI hook in `mir-importer
translator/types.rs`), so:

- The access is a single unwrapped `load <N x T>` / `store <N x T>`, which the
  NVPTX backend fuses into wide transactions — not N scalar `ld`/`st`.
- A vector type's ABI alignment **is** its width, so the emitted `align N` is
  *true* (e.g. `<4 x float>` → `align 16`), not a stamped-on false promise.
  This is the sound counterpart to alignment-annotating a bare `[f32;4]`.

The `scalar_f32x4` kernel uses a plain `#[repr(C)]` array wrapper — the
struct-wrapped `{ [4 x float] }` shape that `CuSimd<f32,4>` produces — and is
the contrast: it stays scalar (4× `ld.b32`) because the aggregate wrapper blocks
the vectorizer regardless of alignment.

## Build & test

```bash
cargo oxide build simd_vec_ops      # emits simd_vec_ops.ptx (no GPU needed)
./analyze.sh                        # pass/fail per kernel; exit code = #failures
```

## Results (sm_80, NVPTX back-end / LLVM 18)

| kernel | type | bits | PTX load/store | result |
|---|---|---|---|---|
| `v_f32x2` | `<2 x f32>` | 64 | `ld.b64` / `st.b64` | 1 transaction |
| `v_f32x4` | `<4 x f32>` | 128 | `ld.v2.b64` / `st.v2.b64` | **v** |
| `v_u32x2` / `v_i32x2` | `<2 x i32>` | 64 | `ld.v2.b32` / `st.v2.b32` | **v2** |
| `v_u32x4` / `v_i32x4` | `<4 x i32>` | 128 | `ld.v4.b32` / `st.v4.b32` | **v4** |
| `v_f64x2` | `<2 x f64>` | 128 | `ld.v2.b64` / `st.v2.b64` | **v2** |
| `v_u64x2` / `v_i64x2` | `<2 x i64>` | 128 | `ld.v2.b64` / `st.v2.b64` | **v2** |
| `v_u16x2` / `v_i16x2` | `<2 x i16>` | 32 | `ld.b32` / `st.b32` | 1 transaction |
| `v_u16x4` / `v_i16x4` | `<4 x i16>` | 64 | `ld.v2.b32` / `st.v2.b32` | **v2** |
| `v_u16x8` / `v_i16x8` | `<8 x i16>` | 128 | `ld.v4.b32` / `st.v4.b32` | **v4** |
| `v_f32x8` | `<8 x f32>` | 256 | `2× ld.v2.b64` | 2 vectors (split) |
| `v_f64x4` | `<4 x f64>` | 256 | `2× ld.v2.b64` | 2 vectors (split) |
| `scalar_f32x4` | `{ [4 x f32] }` | 128 | `4× ld.b32` / `4× st.b32` | **scalar** ✗ |

Every vector kernel collapses the whole vector into 1–2 wide transactions; the
128-bit ones (`v_f32x4`, `v_u32x4`, `v_u16x8`, …) become true `.v2`/`.v4`
instructions, which at SASS are a single 128-bit `LD.E.128` / `ST.E.128`. The 2-
and small-lane cases (`v_f32x2` → `ld.b64`, `v_u16x2` → `ld.b32`) coalesce into
one register-width transaction. Only the `#[repr(C)]` struct-wrapped contrast
stays scalar.

## Why a vector type (and not just alignment)

- A *bare* `[f32;4]` store **does** vectorize given a genuine `align 16` — but a
  `[f32;4]`'s type is only 4-aligned, so claiming 16 is a false promise (it
  faults unless the allocator happens to over-align). Making it truly 16-aligned
  requires a `#[repr(align(16))]` wrapper, which re-introduces the struct layer
  that blocks the vectorizer (`scalar_f32x4` above).
- A vector type `<N x T>` is the one representation that is *both* genuinely
  width-aligned (sound) *and* unwrapped (vectorizes). That's why these kernels
  get sound vector codegen with no alignment hacks.

Routing `CuSimd<T,N>`'s `data` field through this same Vector-ABI path (while
keeping the `tcgen05` construct/extract and lane accessors working) is the
follow-up to make `CuSimd` itself vectorize.
