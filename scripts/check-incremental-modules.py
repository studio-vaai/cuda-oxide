#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU regression and timing checks on the small module fixture.

Uses persistent Cargo/NVIDIA caches. Source is restored even on failures.
Run with CUDA_OXIDE_BACKEND pointing at the matching fork backend.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cargo-oxide', default=shutil.which('cargo-oxide'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/oxide-incremental-modules'))
    parser.add_argument('--alternate-nvjitlink', type=Path, help='Different nvJitLink library for unchanged-source compiler invalidation checks')
    args = parser.parse_args()
    if not args.cargo_oxide:
        parser.error('cargo-oxide must be built/installed')
    root = Path(__file__).resolve().parents[1] / 'crates/rustc-codegen-cuda/examples/incremental_modules'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('CUDA_OXIDE_INCREMENTAL_MODULES', None)
    env.pop('CUDA_OXIDE_PTX_DIR', None)
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version=1', '--no-deps', '--manifest-path', str(root / 'Cargo.toml')], env=env, cwd=root))
    artifact_root = Path(metadata['target_directory']) / 'release/oxide'
    command = [args.cargo_oxide, 'build', '--emit-nvvm-ir', '--', '--release', '--manifest-path', str(root / 'Cargo.toml')]
    report = []

    def build(label, extra_args=()):
        start = time.perf_counter()
        log = output / (label + '.log')
        with log.open('w') as stream:
            code = subprocess.call(command + list(extra_args), env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        entry = {'label': label, 'seconds': time.perf_counter() - start, 'exit_code': code}
        report.append(entry)
        print(json.dumps(entry), flush=True)
        if code:
            print(log.read_text()[-6000:])
            raise RuntimeError(f'{label} failed; see {log}')
        manifest = json.loads((artifact_root / 'incremental_modules.modules.json').read_text())
        (output / (label + '.manifest.json')).write_text(json.dumps(manifest, indent=2))
        return manifest

    def run(arguments=()):
        # Use Cargo metadata: CARGO_TARGET_DIR may be provided by remote compute.
        metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version=1', '--no-deps',
            '--manifest-path', str(root / 'Cargo.toml')], env=env, cwd=root))
        binary = str(Path(metadata['target_directory']) / 'release/incremental_modules')
        if 'CUDA_OXIDE_MODULE_FILES_ONLY' not in env:
            subprocess.run([binary, *arguments], env=env, check=True)
        subprocess.run([binary, '--files', *arguments], env=env, check=True)

    source = root / 'src/lib.rs'
    original = source.read_text()
    cargo_manifest = root / "Cargo.toml"
    original_cargo_manifest = cargo_manifest.read_bytes()
    # Fresh semantic values distinguish a rebuild from an earlier valid cache hit.
    delta = 100 + time.time_ns() % 1000000
    try:
        baseline = build('baseline')
        run()
        assert original.count('kernel_bias = 10u32') == 1
        source.write_text(original.replace('kernel_bias = 10u32', f'kernel_bias = {10 + delta}u32', 1)
            .replace('i as u32 * 4 + 12', f'i as u32 * 4 + {12 + delta}', 1))
        edit = build('kernel_edit')
        run()
        units = {unit['module']: unit for unit in edit['compilation_units']}
        assert not units['incremental_modules::first']['nvvm_hit']
        assert units['incremental_modules::second']['nvvm_hit']
        assert units['incremental_modules::shared']['nvvm_hit']
        assert edit['modules']['incremental_modules::second']['sha256'] == baseline['modules']['incremental_modules::second']['sha256']
        # A helper in a kernel-bearing module must not bring that module's
        # unrelated kernel entries into another module's native link.
        assert 'incremental_modules::first' not in baseline['modules']['incremental_modules::second']['inputs']
        source.write_text(original)
        build('restore')
        source.write_text(original.replace('x: value + 2', f'x: value + {2 + delta}', 1)
            .replace('i as u32 * 4 + 12', f'i as u32 * 4 + {12 + delta}', 1)
            .replace('i as u32 * 6 + 22', f'i as u32 * 6 + {22 + delta}', 1))
        helper = build('shared_helper_edit')
        run()
        units = {unit['module']: unit for unit in helper['compilation_units']}
        assert not units['incremental_modules::shared']['nvvm_hit']
        assert units['incremental_modules::first']['nvvm_hit']
        assert units['incremental_modules::second']['nvvm_hit']
        for name in baseline['modules']:
            assert helper['modules'][name]['sha256'] != baseline['modules'][name]['sha256']
        source.write_text(original.replace('value + 777', f'value + {777 + delta}', 1))
        selective = build('one_consumer_helper_edit')
        run([f'--second-delta={delta}'])
        units = {unit['module']: unit for unit in selective['compilation_units']}
        assert not units['incremental_modules::shared']['nvvm_hit']
        assert units['incremental_modules::first']['nvvm_hit']
        assert units['incremental_modules::second']['nvvm_hit']
        assert selective['modules']['incremental_modules::first']['link_hit']
        assert not selective['modules']['incremental_modules::second']['link_hit']
        assert selective['modules']['incremental_modules::first']['sha256'] == baseline['modules']['incremental_modules::first']['sha256']
        # Preserve the baseline LTO input ordering while omitting only an
        # unrelated native relink. The changed helper still reaches the GPU.
        for name in baseline['modules']:
            assert selective['modules'][name]['inputs'] == baseline['modules'][name]['inputs']
        source.write_text(original.replace('value + 999', 'value + 998', 1))
        unused = build('unreachable_helper_edit')
        assert all(unit['nvvm_hit'] for unit in unused['compilation_units'])
        assert len(unused['modules']) == 2
        source.write_text(original.replace('[1u32, 3, 5]', f'[{1 + delta}u32, 3, 5]'))
        promoted = build('promoted_constant_edit')
        run()
        units = {unit['module']: unit for unit in promoted['compilation_units']}
        assert not units['incremental_modules::first']['nvvm_hit']
        assert units['incremental_modules::second']['nvvm_hit']
        assert promoted['modules']['incremental_modules::second']['sha256'] == baseline['modules']['incremental_modules::second']['sha256']
        # Published immutable files must be repaired from validated cache bytes.
        image = artifact_root / promoted['modules']['incremental_modules::first']['path']
        expected = image.read_bytes()
        image.write_bytes(expected[:-1] + bytes([expected[-1] ^ 1]))
        source.write_text(source.read_text().replace('value + 999', 'value + 997', 1))
        repaired = build('repair_corrupted_published_cubin')
        assert image.read_bytes() == expected
        assert all(m['link_hit'] for m in repaired['modules'].values())
        run()
        # Cargo may select an older Rust library without invoking the backend.
        # Its GPU manifest must follow that exact cached feature configuration.
        cargo_manifest.write_bytes(original_cargo_manifest + b'\n[features]\nmanifest_switch = []\n')
        selected = build('manifest_default_seed')
        alternate = build('manifest_feature_seed', ['--features', 'manifest_switch'])
        assert alternate != selected
        restored = build('manifest_default_fresh', ['--message-format=json-render-diagnostics'])
        assert restored == selected
        events = []
        for line in (output / 'manifest_default_fresh.log').read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get('reason') == 'compiler-artifact' and event['target']['name'] == 'incremental_modules':
                events.append(event)
        assert events and all(event['fresh'] for event in events)
        run()
        # A wrapper/compiler fence can change while the exact LTOIR stays
        # unchanged. Retain the native image produced by the fresh kernel edit
        # instead of asking nvJitLink to optimize that same program again.
        source.write_text(original.replace('kernel_bias = 10u32', f'kernel_bias = {10 + delta}u32', 1)
            .replace('i as u32 * 4 + 12', f'i as u32 * 4 + {12 + delta}', 1))
        old_fence = env.get('CUDA_OXIDE_EXACT_CACHE_TEST_FENCE')
        try:
            env['CUDA_OXIDE_EXACT_CACHE_TEST_FENCE'] = 'first'
            exact_seed = build('exact_input_fence_seed')
            run()
            env['CUDA_OXIDE_EXACT_CACHE_TEST_FENCE'] = 'second'
            exact_reuse = build('exact_input_fence_reuse')
            assert {u['module']: u['ltoir_cache_key'] for u in exact_seed['compilation_units']} == {
                u['module']: u['ltoir_cache_key'] for u in exact_reuse['compilation_units']}
            assert all(m['link_hit'] for m in exact_reuse['modules'].values())
            assert {n: m['sha256'] for n, m in exact_seed['modules'].items()} == {
                n: m['sha256'] for n, m in exact_reuse['modules'].items()}
            run()
        finally:
            if old_fence is None:
                env.pop('CUDA_OXIDE_EXACT_CACHE_TEST_FENCE', None)
            else:
                env['CUDA_OXIDE_EXACT_CACHE_TEST_FENCE'] = old_fence
        if args.alternate_nvjitlink:
            # Only switch the linker. Source, Cargo features and CUDA_* flags
            # stay identical, so Cargo must track the discovered tool identity.
            source.write_text(original)
            selected = build('native_tool_original', ['--message-format=json-render-diagnostics'])
            old_linker = env.get('LIBNVJITLINK_PATH')
            try:
                env['LIBNVJITLINK_PATH'] = str(args.alternate_nvjitlink.resolve(strict=True))
                alternate = build('native_tool_alternate', ['--message-format=json-render-diagnostics'])
                assert alternate != selected, 'Cargo reused a manifest from the old native compiler'
                assert {u['module']: u['nvvm_cache_key'] for u in alternate['compilation_units']} != {
                    u['module']: u['nvvm_cache_key'] for u in selected['compilation_units']}
                run()
                fresh = build('native_tool_alternate_fresh', ['--message-format=json-render-diagnostics'])
                assert fresh == alternate
                events = []
                for line in (output / 'native_tool_alternate_fresh.log').read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get('reason') == 'compiler-artifact' and event['target']['name'] == 'incremental_modules':
                        events.append(event)
                assert events and all(event['fresh'] for event in events)
                run()
            finally:
                if old_linker is None:
                    env.pop('LIBNVJITLINK_PATH', None)
                else:
                    env['LIBNVJITLINK_PATH'] = old_linker
            restored = build('native_tool_restore')
            assert {n: m['sha256'] for n, m in restored['modules'].items()} == {
                n: m['sha256'] for n, m in selected['modules'].items()}
            run()
        print('PASS: isolated kernel cache, generic aggregate device ABI, shared static identity, promoted constants, helper invalidation, unreachable pruning')
    finally:
        cargo_manifest.write_bytes(original_cargo_manifest)
        source.write_text(original)
        build('final_restore')
        (output / 'measurements.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
