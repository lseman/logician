#!/usr/bin/env python3
"""Convert Jupyter notebooks to/from Python scripts and markdown."""
import argparse, json, sys, uuid
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


def nb_to_py(nb):
    """Extract code cells into a Python script."""
    lines = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            lines.append(f'### Cell {i}')
            src = ''.join(cell.get('source', []))
            lines.append(src.rstrip('\n'))
            lines.append('')  # blank line between cells
    return '\n'.join(lines)


def nb_to_md(nb):
    """Convert notebook to markdown with code blocks."""
    parts = []
    # Notebook metadata as info
    if nb.get('metadata', {}).get('kernelspec'):
        ks = nb['metadata']['kernelspec']
        parts.append(f"*Kernel: {ks.get('display_name', '?')}*\n")

    for i, cell in enumerate(nb['cells']):
        ct = cell['cell_type']
        cid = cell.get('id', '?')[:8]

        if ct == 'markdown':
            src = ''.join(cell.get('source', []))
            parts.append(src)
            parts.append('')

        elif ct == 'code':
            src = ''.join(cell.get('source', []))
            parts.append(f'```python')
            parts.append(src.rstrip('\n'))
            parts.append('```')

            # Include outputs
            for o in cell.get('outputs', []):
                otype = o.get('output_type', '?')
                if otype == 'stream':
                    text = o.get('text', '')
                    if text.strip():
                        parts.append(f'<!-- output: {o.get("name", "stdout")} -->')
                        parts.append(text.rstrip())
                        parts.append('')
                elif otype == 'execute_result':
                    data = o.get('data', {})
                    if 'text/plain' in data:
                        parts.append(f'<!-- result -->\n{data["text/plain"]}\n')
                    elif 'text/markdown' in data:
                        parts.append(f'<!-- result (markdown) -->\n{data["text/markdown"]}')
                elif otype == 'error':
                    name = o.get('ename', 'Error')
                    val = o.get('evalue', '')
                    parts.append(f'<!-- error: {name}: {val} -->')

            parts.append('')

    return '\n'.join(parts)


def py_to_nb(py_path, title=None, kernel='python3'):
    """Create notebook from Python script."""
    p = Path(py_path)
    if not p.exists():
        print(f"Error: {py_path} not found", file=sys.stderr)
        sys.exit(1)
    py_text = p.read_text()

    cells = []

    # Title cell
    if title:
        cells.append({
            'cell_type': 'markdown',
            'id': str(uuid.uuid4()),
            'metadata': {},
            'source': [f'# {title}\n']
        })

    # Split script into code cells by '### Cell N' markers or blank-line blocks
    lines = py_text.split('\n')
    current_lines = []
    cell_count = 0

    for line in lines:
        if line.startswith('### Cell'):
            # Save previous cell
            if current_lines:
                cells.append({
                    'cell_type': 'code',
                    'id': str(uuid.uuid4()),
                    'metadata': {},
                    'source': [l + '\n' for l in current_lines[:-1]] + ([current_lines[-1]] if current_lines[-1] else [])
                })
                cell_count += 1
                current_lines = []
            continue
        current_lines.append(line)

    # Last cell
    if current_lines:
        cells.append({
            'cell_type': 'code',
            'id': str(uuid.uuid4()),
            'metadata': {},
            'source': [l + '\n' for l in current_lines[:-1]] + ([current_lines[-1]] if current_lines[-1] else [])
        })

    if not cells:
        # Treat entire file as one code cell
        cells.append({
            'cell_type': 'code',
            'id': str(uuid.uuid4()),
            'metadata': {},
            'source': [l + '\n' for l in py_text.split('\n')[:-1]] + ([py_text.split('\n')[-1]] if py_text.split('\n')[-1] else [])
        })

    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': f'Python {kernel}',
                'language': 'python',
                'name': kernel
            },
            'language_info': {
                'name': 'python',
                'version': '3.11.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }
    return nb


def main():
    parser = argparse.ArgumentParser(description='Convert Jupyter notebooks')
    parser.add_argument('notebook', nargs='?', help='Notebook file')
    parser.add_argument('--to-py', action='store_true', help='Convert to Python script')
    parser.add_argument('--to-md', action='store_true', help='Convert to markdown')
    parser.add_argument('--from-py', metavar='FILE', help='Create notebook from Python script')
    parser.add_argument('--out', '-o', help='Output file (default: stdout)')
    parser.add_argument('--title', help='Title for notebook (used with --from-py)')
    parser.add_argument('--kernel', default='python3', help='Kernel name (default: python3)')
    args = parser.parse_args()

    if args.to_py or args.to_md:
        if not args.notebook:
            print("Error: notebook file required for --to-py / --to-md", file=sys.stderr)
            sys.exit(1)
        nb = load_nb(args.notebook)

        if args.to_py:
            output = nb_to_py(nb)
        else:
            output = nb_to_md(nb)

        if args.out:
            Path(args.out).write_text(output, encoding='utf-8')
            print(f"Wrote {args.out}", file=sys.stderr)
        else:
            print(output)

    elif args.from_py:
        nb = py_to_nb(args.from_py, title=args.title, kernel=args.kernel)
        output = json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
        if args.out:
            Path(args.out).write_text(output, encoding='utf-8')
            print(f"Created {args.out} ({len(nb['cells'])} cells)", file=sys.stderr)
        else:
            print(output)
    else:
        print("Error: specify --to-py, --to-md, or --from-py", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
