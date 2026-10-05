# Incremental native device modules

The default `cargo oxide build -- --release --lib --locked`
workflow partitions reachable device code by `#[cuda_module]`, caches NVVM IR and
LTOIR for each compilation unit, and links architecture-specific cubins with
nvJitLink at build time. The local GPU architecture is detected automatically. For cross-compilation,
set `--arch` or `CUDA_OXIDE_TARGET` to the deployment GPU architecture.
Use `--no-incremental-modules` (or `CUDA_OXIDE_INCREMENTAL_MODULES=0`) to opt out.
The old `--incremental-modules` switch remains accepted.

Independent native links run concurrently by default, bounded by Cargo jobserver
tokens and the available CPU count, up to sixteen workers.
`CUDA_OXIDE_LINK_JOBS=1` selects serial linking; changing this scheduling limit
preserves compiler settings and device cache identity.
The manifest is published only after every link succeeds.

Keep Cargo's target directory between builds. Device files and caches live in
`target/<profile>/oxide/`, honoring Cargo's target-directory configuration and
`CARGO_TARGET_DIR`; no artifact-directory configuration is needed. Cargo host
incrementality is enabled by default; `CARGO_INCREMENTAL=0` opts out. Use a stable `codegen-units` setting across builds. The Studio iteration
profile uses 256 units and disables debug information. The wrapper explicitly enables release MIR
inlining because rustc's incremental default would disable it.

Device-only modules emit intermediate code only when reachable. A helper called
across modules is an external declaration in its caller's unit and a definition
in its own unit. The final cubin links the caller's transitive LTOIR dependencies,
so each cubin is independently loadable. Helpers in a module that also contains
kernels use a separate dependency unit, avoiding retention of unrelated entries.
Editing a helper can therefore reuse caller NVVM/LTOIR while relinking affected
cubins. Rust-inlined helpers naturally invalidate the callers containing them.

A kernel-bearing module gets one native artifact. Large modules can still take a
long time to relink; splitting those modules is an application-level optimization.
Cold native builds can cost more than a single PTX build. This feature targets
repeated edits and removes device compilation from application startup.

Cache keys track MIR, promoted constants, declaration ABI, compiler settings,
CUDA tools and target. Local type/constant changes and dependency metadata changes
conservatively invalidate all units. Debug builds retain source-position hashing;
position-independent builds ignore spans. Shared globals use stable symbol
identity and dynamic shared helpers honor the strongest alignment contract.
Cache payloads and published cubins carry SHA-256 checksums.

The emitted `<crate>.modules.json` is the publication boundary. Ship it with exactly
its referenced `<crate>.modules/<digest>.cubin` files, or use the embedded bundles
in the Rust executable. The runtime selects the logical module's native image,
validates file digests and architecture compatibility, and loads it directly.
Load/preload modules before CUDA stream capture. Native-only builds require a
compatible GPU; there is no PTX fallback in this mode.

Ordinary device globals shared across cubins and cross-crate generic-kernel bundle
merging are currently rejected. Use explicit device buffers or `--no-incremental-modules` for those programs.
Diagnostic commands that explicitly inspect PTX/IR retain their requested format.

## Small GPU regression fixture

Build the backend with the repository's pinned nightly, then run:

```sh
CUDA_OXIDE_BACKEND=/absolute/path/librustc_codegen_cuda.so \
  CARGO_PROFILE_RELEASE_INCREMENTAL=true \
  CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  python3 scripts/check-incremental-modules.py --cargo-oxide /absolute/path/cargo-oxide
```

The fixture checks native launches, embedded/file loading, generic aggregate ABI,
shared static identity, dynamic shared alignment, promoted constants, helper
invalidation, unreachable pruning, and isolation of an unrelated kernel module.

## Kernel body edits with an existing host executable

Use file loading and seed a full release library with the same compiler, target,
features, dependencies and profile:

```sh
export CUDA_OXIDE_MODULE_FILES_ONLY=1
export CUDA_OXIDE_REUSE_HOST_FOR_KERNEL_EDITS=1
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_INCREMENTAL=true
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256
cargo oxide build --emit-nvvm-ir -- --release --lib --locked
# Build the application's host executable once, using the generated files.
cargo oxide --kernels-only my-kernel-crate build --emit-nvvm-ir -- --release --lib --locked
# Restart that existing executable to load the new cubins.
```

The fast command retains the host rlib, metadata and executable. Rust parses and
checks the selected crate, walks only local non-generic kernel roots, and emits
no Rust metadata or host LLVM code. It avoids collecting the CPU monomorphization
graph. A source
contract hashes host/helper HIR bodies (including closures), attributes, types,
ABI, observable source locations, compiler options and dependency identities.
Only reserved CUDA kernel entry bodies are excluded. CPU/helper/ABI/location
changes, CPU references to raw kernel entries, changed dependencies, and a damaged
host proof cache reject the update before publishing a new native manifest. Run a
normal full build after such changes. This mode requires debug information off;
request device line tables explicitly when debugging, using a full build.

A persistent `target/oxide-kernels-only` cache is seeded from the normal release
cache once per host contract. The initial copy is setup cost. Subsequent edits
reuse it. A normal build after a kernel-only update refreshes the selected root
library even when the source has returned to its original contents; dependency
and device caches stay intact. Cache locks still require one build per target at
a time. Restart the process after updates: loaded module caches are immutable.

Run `scripts/check-kernels-only.py` and `scripts/check-host-kernel-reuse.py` with
`--cargo-oxide PATH --output DIRECTORY` to test actual launches, frozen host
artifacts, repeated edits, source-location/closure/ABI rejection, cache corruption,
and returning to a full build. Native cubin digests hash each LTOIR input once;
unchanged published cubins are verified and retained instead of rewritten.
