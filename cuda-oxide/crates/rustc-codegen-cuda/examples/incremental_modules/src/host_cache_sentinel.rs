// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Keep a CPU body in a separate source file so kernel edits can leave its CGU
// green even when spans in the edited file acquire a new source checksum.
#[inline(never)]
pub fn value() -> u32 {
    std::hint::black_box(37)
}
