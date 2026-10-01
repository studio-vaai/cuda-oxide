#!/usr/bin/env python3
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
    args = parser.parse_args()
    if not args.cargo_oxide:
        parser.error('cargo-oxide must be built/installed')
    root = Path(__file__).resolve().parents[1] / 'crates/rustc-codegen-cuda/examples/incremental_modules'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_OXIDE_INCREMENTAL_MODULES'] = '1'
    env['CUDA_OXIDE_PTX_DIR'] = str(root)
    command = [args.cargo_oxide, 'build', '--emit-nvvm-ir', '--', '--release', '--manifest-path', str(root / 'Cargo.toml')]
    report = []

    def build(label):
        start = time.perf_counter()
        log = output / (label + '.log')
        with log.open('w') as stream:
            code = subprocess.call(command, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        entry = {'label': label, 'seconds': time.perf_counter() - start, 'exit_code': code}
        report.append(entry)
        print(json.dumps(entry), flush=True)
        if code:
            print(log.read_text()[-6000:])
            raise RuntimeError(f'{label} failed; see {log}')
        manifest = json.loads((root / 'incremental_modules.modules.json').read_text())
        (output / (label + '.manifest.json')).write_text(json.dumps(manifest, indent=2))
        return manifest

    def run():
        # Use Cargo metadata: CARGO_TARGET_DIR may be provided by remote compute.
        metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version=1', '--no-deps',
            '--manifest-path', str(root / 'Cargo.toml')], env=env, cwd=root))
        binary = str(Path(metadata['target_directory']) / 'release/incremental_modules')
        if 'CUDA_OXIDE_MODULE_FILES_ONLY' not in env:
            subprocess.run([binary], env=env, check=True)
        subprocess.run([binary, '--files'], env=env, check=True)

    source = root / 'src/lib.rs'
    original = source.read_text()
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
        image = root / promoted['modules']['incremental_modules::first']['path']
        expected = image.read_bytes()
        image.write_bytes(expected[:-1] + bytes([expected[-1] ^ 1]))
        source.write_text(source.read_text().replace('value + 999', 'value + 997', 1))
        repaired = build('repair_corrupted_published_cubin')
        assert image.read_bytes() == expected
        assert all(m['link_hit'] for m in repaired['modules'].values())
        run()
        print('PASS: isolated kernel cache, generic aggregate device ABI, shared static identity, promoted constants, helper invalidation, unreachable pruning')
    finally:
        source.write_text(original)
        build('final_restore')
        (output / 'measurements.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
