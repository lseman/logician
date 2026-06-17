#!/usr/bin/env python3
"""Run Jupyter notebook cells."""
import argparse, json, sys, os, re, shutil, subprocess, tempfile
from pathlib import Path


def load_nb(path):
    p = Path(path)
    if not p.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON — {e}", file=sys.stderr)
        sys.exit(1)


def save_nb(nb, path):
    p = Path(path)
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


def parse_cell_spec(spec, n):
    """Parse '0,2,4-6' into list of indices. 'all' = all cells."""
    if spec.strip().lower() == 'all':
        return list(range(n))
    indices = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            start, end = int(start), int(end)
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    # Deduplicate, preserve order, sort
    seen = set()
    result = []
    for i in indices:
        if i < 0 or i >= n:
            print(f"Error: cell index {i} out of range (0-{n-1})", file=sys.stderr)
            sys.exit(1)
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


def make_python_script(cell):
    """Convert cell source to a Python script (for fallback execution)."""
    src = ''.join(cell.get('source', []))
    lines = []
    for line in src.split('\n'):
        # Remove IPython magic handling warning
        lines.append(line)
    return '\n'.join(lines) + '\n'


def try_jupyter_client(nb, cell_indices, timeout, sleep_interval, quiet, cwd):
    """Try running via jupyter_client."""
    try:
        from jupyter_client import BlockingKernelClient
        from jupyter_client.manager import start_new_kernel
    except ImportError:
        return False

    print("Using jupyter_client to run cells...", file=sys.stderr)
    kernel_proc, kc = start_new_kernel(cwd=str(cwd))
    try:
        kc.wait_for_ready(timeout=60)
        for i in cell_indices:
            cell = nb['cells'][i]
            src = ''.join(cell.get('source', []))
            if not src.strip():
                continue
            if not quiet:
                print(f"\n--- Running cell {i} ---", file=sys.stderr)
                print(src, end='', file=sys.stderr)

            msg_id = kc.execute(src)
            outputs = []
            try:
                while True:
                    msg = kc.get_shell_msg(timeout=timeout)
                    msg_type = msg['msg_type']
                    content = msg['content']

                    if msg_type == 'execute_input':
                        pass
                    elif msg_type == 'execute_result':
                        outputs.append({
                            'output_type': 'execute_result',
                            'execution_count': content.get('execution_count'),
                            'data': content.get('data', {}),
                            'metadata': content.get('metadata', {})
                        })
                    elif msg_type == 'display_data':
                        outputs.append({
                            'output_type': 'display_data',
                            'data': content.get('data', {}),
                            'metadata': content.get('metadata', {})
                        })
                    elif msg_type == 'stream':
                        outputs.append({
                            'output_type': 'stream',
                            'name': content.get('name', 'stdout'),
                            'text': content.get('text', '')
                        })
                    elif msg_type == 'error':
                        outputs.append({
                            'output_type': 'error',
                            'ename': content.get('ename', 'Error'),
                            'evalue': content.get('evalue', ''),
                            'traceback': content.get('traceback', [])
                        })
                    elif msg_type == 'execute_reply' and content.get('status') == 'ok':
                        break
                    elif msg_type == 'execute_reply':
                        if not outputs:
                            outputs.append({
                                'output_type': 'error',
                                'ename': 'ExecutionError',
                                'evalue': content.get('status', 'failed'),
                                'traceback': []
                            })
                        break
            except Exception:
                if not outputs:
                    outputs.append({
                        'output_type': 'error',
                        'ename': 'Timeout',
                        'evalue': f'Cell {i} timed out after {timeout}s',
                        'traceback': []
                    })

            nb['cells'][i]['outputs'] = outputs
            nb['cells'][i]['execution_count'] = 1 if outputs else None

            if sleep_interval > 0 and i < cell_indices[-1]:
                print(f"Sleeping {sleep_interval}s for interactivity...", file=sys.stderr)
                time.sleep(sleep_interval)
    finally:
        try:
            kc.shutdown()
        except Exception:
            pass
        try:
            kernel_proc.kill()
        except Exception:
            pass
    return True


def try_fallback(nb, cell_indices, timeout, sleep_interval, quiet, cwd):
    """Run cells by writing each to a temp file and executing with python."""
    print("No jupyter_client available. Running cells via subprocess (python).", file=sys.stderr)

    for i in cell_indices:
        cell = nb['cells'][i]
        src = ''.join(cell.get('source', []))
        if not src.strip():
            continue

        if not quiet:
            print(f"\n--- Running cell {i} ---", file=sys.stderr)
            print(src, end='', file=sys.stderr)

        # Write cell to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=str(cwd)) as f:
            f.write(src)
            tmp_path = f.name

        # Capture stdout/stderr
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout, cwd=str(cwd)
        )

        outputs = []
        if proc.returncode == 0:
            if proc.stdout:
                outputs.append({
                    'output_type': 'stream',
                    'name': 'stdout',
                    'text': proc.stdout
                })
            if proc.stderr:
                outputs.append({
                    'output_type': 'stream',
                    'name': 'stderr',
                    'text': proc.stderr
                })
        else:
            # Build error output
            err_name = 'SubprocessError'
            err_value = proc.stderr.strip().split('\n')[-1] if proc.stderr else 'exit code ' + str(proc.returncode)
            tb = proc.stderr.strip().split('\n') if proc.stderr else []
            outputs.append({
                'output_type': 'error',
                'ename': err_name,
                'evalue': err_value,
                'traceback': tb
            })

        nb['cells'][i]['outputs'] = outputs
        nb['cells'][i]['execution_count'] = 1 if outputs else None

        # Remove temp file
        os.unlink(tmp_path)

        if sleep_interval > 0 and i < cell_indices[-1]:
            print(f"Sleeping {sleep_interval}s for interactivity...", file=sys.stderr)
            time.sleep(sleep_interval)


def time(name, fn):
    """Simple timing wrapper."""
    import time
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s", file=sys.stderr)
    return result


def main():
    import time

    parser = argparse.ArgumentParser(description='Run Jupyter notebook cells')
    parser.add_argument('notebook')
    parser.add_argument('--cells', default='all', help='Cell indices: "all", "0,2,4", "0-5" (default: all)')
    parser.add_argument('--timeout', '-t', type=int, default=60, help='Timeout per cell in seconds (default: 60)')
    parser.add_argument('--quiet', action='store_true', help='Suppress terminal output')
    parser.add_argument('--sleep', '-i', type=float, default=0, help='Sleep between cells for interactivity (seconds)')
    parser.add_argument('--method', choices=['auto', 'jupyter', 'subprocess'], default='auto',
                        help='Execution method (default: auto)')
    args = parser.parse_args()

    nb = load_nb(args.notebook)
    p = Path(args.notebook)
    shutil.copy2(str(p), str(p) + '.bak')

    cwd = p.parent
    indices = parse_cell_spec(args.cells, len(nb['cells']))

    if not indices:
        print("No cells to run.")
        return

    if not args.quiet:
        print(f"Running {len(indices)} cells: {indices}", file=sys.stderr)
        print(f"Notebook: {p.name}", file=sys.stderr)

    if args.method == 'jupyter':
        try_jupyter_client(nb, indices, args.timeout, args.sleep, args.quiet, cwd)
    elif args.method == 'subprocess':
        try_fallback(nb, indices, args.timeout, args.sleep, args.quiet, cwd)
    else:
        # Auto: try jupyter first, fall back to subprocess
        if not try_jupyter_client(nb, indices, args.timeout, args.sleep, args.quiet, cwd):
            try_fallback(nb, indices, args.timeout, args.sleep, args.quiet, cwd)

    save_nb(nb, args.notebook)
    if not args.quiet:
        print(f"\nDone. Updated {p.name}", file=sys.stderr)


if __name__ == '__main__':
    main()
