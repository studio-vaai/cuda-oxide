// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[test]
fn host_stubs_keep_source_lint_hygiene() {
    unsafe { std::env::set_var(reserved_oxide_symbols::HOST_KERNEL_STUBS_ENV, "1") };
    let tests = trybuild::TestCases::new();
    tests.pass("tests/pass/host_stub_lints.rs");
    tests.compile_fail("tests/compile_fail/host_stub_unused_import.rs");
}
