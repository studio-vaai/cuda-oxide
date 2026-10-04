// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// trybuild's generated crate does not mirror feature flags onto the separate
// cuda-host dev dependency. Async expansion is covered by the macro unit tests
// and cuda-host integration tests instead.
#[cfg(not(feature = "async"))]
#[test]
fn policy_constants_remain_typed_and_fail_closed() {
    // Validate policy checks inside complete device bodies. This binary has
    // one test, and sets the opt-out before trybuild spawns child processes.
    unsafe { std::env::set_var(reserved_oxide_symbols::HOST_KERNEL_STUBS_ENV, "0") };
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/policy_config_expressions.rs");
    t.compile_fail("tests/compile_fail/policy_config_unresolved.rs");
    t.compile_fail("tests/compile_fail/policy_config_zero_threads.rs");
    t.compile_fail("tests/compile_fail/policy_contract_zero_threads.rs");
    t.compile_fail("tests/compile_fail/policy_contract_exact_block_too_large.rs");
    t.compile_fail("tests/compile_fail/policy_contract_large_3d_block.rs");
    t.compile_fail("tests/compile_fail/policy_contract_wrong_brand.rs");
    t.compile_fail("tests/compile_fail/policy_config_invalid_unroll.rs");
}
