# Programmatic Dependent Launch (PDL)

Demonstrates CUDA Programmatic Dependent Launch end to end:

- **Device intrinsics** (`cuda_device::pdl`):
  - `griddepcontrol_launch_dependents()` — the *primary* kernel signals that
    dependent kernels may launch (PTX `griddepcontrol.launch_dependents`,
    CUDA C++ `cudaTriggerProgrammaticLaunchCompletion()`).
  - `griddepcontrol_wait()` — the *secondary* kernel blocks until all
    upstream grids have completed and flushed their global-memory writes
    (PTX `griddepcontrol.wait`, CUDA C++ `cudaGridDependencySynchronize()`).
- **Launch attribute**: the secondary kernel is launched with
  `cuda_launch! { ..., programmatic: true }`, which sets
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` via
  `cuda_core::launch_kernel_programmatic_on_stream`.

## What it shows

The primary kernel writes a buffer, triggers, then spins for 2 ms of "tail"
work. The secondary kernel spins for 2 ms of "prologue" work that does not
touch the primary's output, calls `griddepcontrol_wait()`, then consumes the
buffer. Both kernels timestamp their phases with `%globaltimer`.

- With a normal launch the two kernels serialize: the secondary starts after
  the primary ends.
- With `programmatic: true` the secondary launches while the primary is still
  spinning, so its prologue overlaps the primary's tail — the timestamps show
  the secondary starting ~2 ms *before* the primary ends, and the pair takes
  ~2 ms instead of ~4 ms.

Data correctness is asserted in both modes (the `wait` orders the secondary's
reads after the primary's flushed writes). Overlap itself is reported but not
asserted: PDL concurrency is opportunistic per the CUDA programming model.

## Requirements

Compute capability 9.0+ (Hopper / Blackwell). On older devices the
`griddepcontrol` PTX instructions fail to assemble.

## Run

```
cargo oxide run pdl
```
