# Device convergence

Arithmetic and thread-ID reads do not need LLVM's `convergent` restriction.
A helper that reaches a block barrier or warp shuffle does, as do its transitive
callers. This example keeps two helper boundaries on each path and checks both cases.

```text
arithmetic_leaf -> arithmetic_wrapper -> arithmetic_probe
thread ID -> thread_value_leaf -> thread_value_wrapper -> loop_invariant_probe
barrier intrinsic -> barrier_leaf -> barrier_wrapper -> barrier_probe
shuffle intrinsic -> shuffle_leaf -> shuffle_wrapper -> shuffle_probe
```

The arrows show how convergence requirements propagate from callee to caller.
The first two chains stay unrestricted. The collective chains retain protection.

## Build and run

From the `cuda-oxide/` workspace, with its pinned toolchain and CUDA toolkit:

```bash
cargo oxide run device_convergence
bash crates/rustc-codegen-cuda/examples/device_convergence/verify-code-shape.sh
```

The layout follows the other standalone examples: `Cargo.toml`, `Cargo.lock`,
`src/main.rs`, and an automatically discovered `verify-code-shape.sh`.
The package-scoped MIR-inline flag preserves the helper boundaries for inspection.

## What reproduces the issue

The `loop_invariant_probe` kernel sums an invariant helper result a runtime number
of times. The helper reads `threadIdx.x` and adds a runtime base modulo 2^32:

```rust
let mut total = 0u32;
let mut i = 0u32;
while i < iterations {
    total = total.wrapping_add(thread_value_wrapper(base));
    i += 1;
}
```

The value is constant for each thread, so LLVM can replace the summation with
`thread_value_wrapper(base).wrapping_mul(iterations)` when the loop is nonempty.
Blanket convergence prevents this optimization in the tested pipeline.
After selective inference, the optimized PTX contains one helper call, one
32-bit multiply, and no loop backedge. `verify-code-shape.sh` requires that shape
without depending on register names, label numbers, or instruction scheduling.
This PTX check runs first so unchanged upstream fails on a compiled-code difference.

Measured with Rust 1.99.1 / LLVM 23.1.1 targeting `sm_120`, using the same kernel
source and compiler options on upstream `76ac102166` and with the convergence fix:

| `loop_invariant_probe` | Upstream | With fix |
| --- | ---: | ---: |
| Static helper-call sites | 2 | 1 |
| Loop backedges | 2 | 0 |
| Static PTX instructions | 67 | 42 |

Instruction totals include the PTX call argument/return transfers and output
indexing. They describe PTX code size, not executed SASS or measured runtime.

The script also reads the emitted **pre-optimization** `.ll` file. Arithmetic
and thread-ID helpers must be unrestricted, while collective wrapper chains,
calls, entries, and intrinsic calls stay convergent. PTX must retain all four
entries, the barrier, and the shuffle.

## Runtime checks

Four blocks of 128 threads produce 512 exact integer results per path:

- Arithmetic called from lane-dependent branches.
- A shared-memory neighbor exchange across a block barrier.
- An XOR-1 lane exchange through a full-warp shuffle.
- An invariant summation with runtime loop counts 1, 3, 17, 256, and 0.

The invariant path uses a base near `u32::MAX` to exercise wrapping addition and
multiplication. Its final zero-count launch overwrites earlier nonzero results.
There are 4,096 checked results across all paths.

Every thread reaches the same barrier and shuffle. Blocks contain complete warps;
the output bounds check comes after each collective. No raw inline PTX is used.

Expected output:

```text
arithmetic: PASS (512 exact results)
barrier: PASS (512 exact results)
shuffle: PASS (512 exact results)
loop_invariant(1): PASS (512 exact results)
loop_invariant(3): PASS (512 exact results)
loop_invariant(17): PASS (512 exact results)
loop_invariant(256): PASS (512 exact results)
loop_invariant(0): PASS (512 exact results)
SUCCESS: all 4096 arithmetic, collective, and loop-invariant results match
loop_invariant_probe PTX: PASS (one helper call, one multiply, no loop)
device_convergence code shape: PASS
```

These checks establish attribute selection and numerical correctness. They do
not measure an application speedup.
