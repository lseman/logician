#!/usr/bin/env python3
"""Format/normalize a Jupyter notebook."""
import argparse, json, sys, shutil
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


def format_cell(cell):
    """Normalize a single cell."""
    # Ensure source is list of strings
    src = cell.get('source', [])
    if isinstance(src, str):
        if src:
            cell['source'] = src.split('\n')
            # Add trailing newlines except to last element
            cell['source'] = [l + '\n' for l in cell['source'][:-1]]
            if cell['source'][-1] == '\n':
                cell['source'][-1] = ''
        else:
            cell['source'] = []
    elif isinstance(src, list):
        cell['source'] = [str(s) if not isinstance(s, str) else s for s in src]

    # Strip trailing whitespace from each source line
    cell['source'] = [s.rstrip() for s in cell['source']]

    # Ensure id exists
    if 'id' not in cell:
        import uuid
        cell['id'] = str(uuid.uuid4())

    # Clean metadata — keep known keys
    if 'metadata' in cell and isinstance(cell['metadata'], dict):
        known_keys = {'collapsed', 'execution', 'hide_input', 'trusted', 'scroll',
                      'slideshow', 'tags', 'nbsphinx', 'caption', 'pycharm', 'jupyter'}
        cell['metadata'] = {k: v for k, v in cell['metadata'].items() if k in known_keys}

    # Normalize outputs for code cells
    if cell['cell_type'] == 'code':
        outputs = cell.get('outputs', [])
        if isinstance(outputs, list):
            # Sort by output_type then execution_count
            def sort_key(o):
                return (o.get('output_type', ''), o.get('execution_count', 0))
            cell['outputs'] = sorted(outputs, key=sort_key)

    return cell


def format_nb(nb):
    """Normalize entire notebook."""
    for cell in nb['cells']:
        format_cell(cell)

    # Clean top-level metadata
    if 'metadata' in nb and isinstance(nb['metadata'], dict):
        known_keys = {'kernelspec', 'language_info', 'colab', 'kernelspec',
                       'accelerator', 'widgets', 'toc', 'varInspector'}
        nb['metadata'] = {k: v for k, v in nb['metadata'].items() if k in known_keys}

    # Ensure format fields
    nb.setdefault('nbformat', 4)
    nb.setdefault('nbformat_minor', 5)

    return nb


def main():
    parser = argparse.ArgumentParser(description='Format Jupyter notebook')
    parser.add_argument('notebook')
    parser.add_argument('--check', action='store_true', help='Check formatting without writing')
    parser.add_argument('--indent', type=int, default=2, help='JSON indent (default: 2)')
    args = parser.parse_args()

    nb = load_nb(args.notebook)
    formatted = format_nb(nb)

    formatted_text = json.dumps(formatted, indent=args.indent, ensure_ascii=False) + '\n'
    original_text = Path(args.notebook).read_text()

    if formatted_text == original_text:
        print("Notebook is already formatted.", file=sys.stderr)
        if args.check:
            sys.exit(0)
    else:
        if args.check:
            print("Notebook needs formatting.", file=sys.stderr)
            sys.exit(1)
        # Backup and write
        p = Path(args.notebook)
        shutil.copy2(str(p), str(p) + '.bak')
        p.write_text(formatted_text, encoding='utf-8')
        print(f"Formatted {args.notebook}", file=sys.stderr)


if __name__ == '__main__':
    main()
