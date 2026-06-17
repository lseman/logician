#!/usr/bin/env python3
"""Edit Jupyter notebook cells."""
import argparse, json, sys, uuid, shutil
from pathlib import Path


def load_nb(path, backup=True):
    p = Path(path)
    if not p.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    try:
        nb = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON — {e}", file=sys.stderr)
        sys.exit(1)
    if backup:
        shutil.copy2(str(p), str(p) + '.bak')
    return nb


def save_nb(nb, path):
    p = Path(path)
    # Normalize: ensure source is array of strings
    for cell in nb['cells']:
        src = cell.get('source', [])
        if isinstance(src, str):
            cell['source'] = [src] if src else []
        elif isinstance(src, list):
            # Ensure each element is a string
            cell['source'] = [str(s) if not isinstance(s, str) else s for s in src]
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


def _read_source(args):
    """Read source text from --source-file, --source-stdin, or --source."""
    count = sum([
        args.source_file is not None,
        args.source_stdin,
        args.source is not None
    ])
    if count > 1:
        print("Error: specify only one of --source-file, --source-stdin, or --source", file=sys.stderr)
        sys.exit(1)

    if args.source_file:
        p = Path(args.source_file)
        if not p.exists():
            print(f"Error: source file {args.source_file} not found", file=sys.stderr)
            sys.exit(1)
        return p.read_text()

    if args.source_stdin:
        return sys.stdin.read()

    if args.source is not None:
        return args.source

    print("Error: specify --source, --source-file, or pipe via --source-stdin", file=sys.stderr)
    sys.exit(1)


def _split_lines(source):
    """Convert source string to array of lines (notebook format)."""
    if not source:
        return []
    parts = source.split('\n')
    # Every line except the last gets a trailing newline
    return [l + '\n' for l in parts[:-1]] + ([parts[-1]] if parts else [])


def add_code_cell(nb, source, after=None, before=None):
    cell = {
        'cell_type': 'code',
        'id': str(uuid.uuid4()),
        'metadata': {},
        'source': _split_lines(source)
    }
    n = len(nb['cells'])
    if before is not None:
        idx = min(max(before, 0), n)
    elif after is not None:
        idx = min(max(after + 1, 0), n)
    else:
        idx = n
    nb['cells'].insert(idx, cell)
    return idx


def add_markdown_cell(nb, source, append=False):
    cell = {
        'cell_type': 'markdown',
        'id': str(uuid.uuid4()),
        'metadata': {},
        'source': _split_lines(source)
    }
    if append:
        nb['cells'].append(cell)
        return len(nb['cells']) - 1
    else:
        # Default: insert at end
        nb['cells'].append(cell)
        return len(nb['cells']) - 1


def delete_cell(nb, idx):
    n = len(nb['cells'])
    if idx < 0 or idx >= n:
        print(f"Error: cell index {idx} out of range (0-{n-1})", file=sys.stderr)
        sys.exit(1)
    cell = nb['cells'].pop(idx)
    print(f"Deleted cell {idx} (id={cell.get('id', '?')}, type={cell['cell_type']})")


def replace_cell(nb, idx, source):
    n = len(nb['cells'])
    if idx < 0 or idx >= n:
        print(f"Error: cell index {idx} out of range (0-{n-1})", file=sys.stderr)
        sys.exit(1)
    old_cell = nb['cells'][idx]
    old_id = old_cell.get('id', '?')
    old_len = len(''.join(old_cell.get('source', [])))
    new_len = len(source)
    old_cell['source'] = _split_lines(source)
    print(f"Replaced cell {idx} (id={old_id}): {old_len} -> {new_len} chars")


def move_cell(nb, from_idx, to_idx):
    n = len(nb['cells'])
    if from_idx < 0 or from_idx >= n:
        print(f"Error: source index {from_idx} out of range", file=sys.stderr)
        sys.exit(1)
    if to_idx < 0 or to_idx > n:
        print(f"Error: target index {to_idx} out of range (0-{n})", file=sys.stderr)
        sys.exit(1)
    cell = nb['cells'].pop(from_idx)
    nb['cells'].insert(to_idx, cell)
    print(f"Moved cell {from_idx} -> {to_idx}")


def rename_cell(nb, idx, new_id):
    n = len(nb['cells'])
    if idx < 0 or idx >= n:
        print(f"Error: cell index {idx} out of range", file=sys.stderr)
        sys.exit(1)
    old_id = nb['cells'][idx].get('id', '?')
    nb['cells'][idx]['id'] = new_id
    print(f"Renamed cell {idx}: {old_id} -> {new_id}")


def clear_cell_outputs(nb, idx):
    n = len(nb['cells'])
    if idx < 0 or idx >= n:
        print(f"Error: cell index {idx} out of range", file=sys.stderr)
        sys.exit(1)
    cell = nb['cells'][idx]
    if cell['cell_type'] != 'code':
        print(f"Error: cell {idx} is not a code cell (type={cell['cell_type']})", file=sys.stderr)
        sys.exit(1)
    old_count = len(cell.get('outputs', []))
    cell['outputs'] = []
    cell['execution_count'] = None
    print(f"Cleared outputs from cell {idx} ({old_count} outputs removed)")
    return nb


def clear_all_outputs(nb):
    count = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and cell.get('outputs'):
            cell['outputs'] = []
            cell['execution_count'] = None
            count += 1
    print(f"Cleared outputs from {count} code cells")
    return nb


def main():
    parser = argparse.ArgumentParser(description='Edit Jupyter notebook')
    parser.add_argument('notebook')
    sub = parser.add_subparsers(dest='action', required=True)

    # add-code
    p = sub.add_parser('add-code')
    p.add_argument('--source', help='Inline source text (short content)')
    p.add_argument('--source-file', help='Read source from file')
    p.add_argument('--source-stdin', action='store_true', help='Read source from stdin')
    p.add_argument('--after', type=int, help='Insert after cell index')
    p.add_argument('--before', type=int, help='Insert before cell index')

    # add-markdown
    p = sub.add_parser('add-markdown')
    p.add_argument('--source', help='Inline markdown text')
    p.add_argument('--source-file', help='Read markdown from file')
    p.add_argument('--source-stdin', action='store_true', help='Read markdown from stdin')
    p.add_argument('--append', action='store_true', help='Append at end (default)')

    # delete-cell
    p = sub.add_parser('delete-cell')
    p.add_argument('index', type=int)

    # replace-cell
    p = sub.add_parser('replace-cell')
    p.add_argument('index', type=int)
    p.add_argument('--source', help='Inline source text')
    p.add_argument('--source-file', help='Read source from file')
    p.add_argument('--source-stdin', action='store_true', help='Read source from stdin')

    # move-cell
    p = sub.add_parser('move-cell')
    p.add_argument('--from-cell', type=int, required=True, help='Source cell index')
    p.add_argument('--to-cell', type=int, required=True, help='Target cell index')

    # rename
    p = sub.add_parser('rename')
    p.add_argument('index', type=int)
    p.add_argument('new_id')

    # clear-output
    p = sub.add_parser('clear-output')
    p.add_argument('index', type=int)

    # clear-all-outputs
    sub.add_parser('clear-all-outputs')

    args = parser.parse_args()
    nb = load_nb(args.notebook)

    if args.action in ('add-code', 'add-markdown'):
        source = _read_source(args)
        if args.action == 'add-code':
            idx = add_code_cell(nb, source, after=args.after, before=args.before)
            print(f"Added code cell at index {idx}")
        else:
            idx = add_markdown_cell(nb, source, append=args.append)
            print(f"Added markdown cell at index {idx}")

    elif args.action == 'delete-cell':
        delete_cell(nb, args.index)

    elif args.action == 'replace-cell':
        source = _read_source(args)
        replace_cell(nb, args.index, source)

    elif args.action == 'move-cell':
        move_cell(nb, args.from_cell, args.to_cell)

    elif args.action == 'rename':
        rename_cell(nb, args.index, args.new_id)

    elif args.action == 'clear-output':
        clear_cell_outputs(nb, args.index)

    elif args.action == 'clear-all-outputs':
        clear_all_outputs(nb)

    save_nb(nb, args.notebook)


if __name__ == '__main__':
    main()
