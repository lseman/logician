#!/usr/bin/env python3
"""Inspect a Jupyter notebook."""
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


def summary(nb):
    print(f"Notebook: {Path(nb['file']).name}")
    print(f"Cells: {len(nb['cells'])}  |  nbformat: {nb.get('nbformat', 4)}.{nb.get('nbformat_minor', 5)}")
    if nb.get('metadata', {}).get('kernelspec'):
        ks = nb['metadata']['kernelspec']
        print(f"Kernel: {ks.get('display_name', ks.get('name', '?'))}")
    code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    md_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
    total_src = sum(len(''.join(c.get('source', []))) for c in nb['cells'])
    total_out = sum(len(c.get('outputs', [])) for c in nb['cells'] if c['cell_type'] == 'code')
    print(f"Code: {code_count}  Markdown: {md_count}  Source chars: {total_src}  Output cells: {total_out}")
    print()
    for i, cell in enumerate(nb['cells']):
        ct = cell['cell_type']
        cid = cell.get('id', '?')
        if ct == 'code':
            src = ''.join(cell.get('source', []))
            lines = len(src.splitlines())
            out_count = len(cell.get('outputs', []))
            out_info = f"  +{out_count} outputs" if out_count else ""
            print(f"  [{i:>3}] code   id={cid}  {lines:>4} lines{out_info}")
        else:
            src = ''.join(cell.get('source', []))
            first_line = src.splitlines()[0][:60] if src.splitlines() else ''
            print(f"  [{i:>3}] md     id={cid}  '{first_line}'")


def full_cells(nb):
    for i, cell in enumerate(nb['cells']):
        ct = cell['cell_type']
        cid = cell.get('id', '?')
        src = ''.join(cell.get('source', []))
        print(f"=== Cell {i} (id={cid}, type={ct}) ===")
        if src:
            print(src, end='')
        else:
            print("(empty)")
        if ct == 'code' and cell.get('outputs'):
            print("\n--- OUTPUTS ---")
            for o in cell['outputs']:
                otype = o.get('output_type', '?')
                if otype == 'error':
                    print(f"[ERROR] {o.get('ename', '?')}: {o.get('evalue', '?')}", file=sys.stderr)
                elif otype in ('execute_result', 'display_data'):
                    data = o.get('data', {})
                    for mime, content in sorted(data.items()):
                        if mime == 'text/plain':
                            print(content, end='')
                        elif mime == 'text/markdown':
                            print(content, end='')
                        elif mime == 'text/html':
                            print(f"[HTML output: {len(content)} bytes]", file=sys.stderr)
                        elif mime.startswith('image/'):
                            print(f"[{mime} output: {len(content)} bytes]", file=sys.stderr)
                        else:
                            print(f"[{mime}: {len(str(content))} chars]", file=sys.stderr)
                elif otype == 'stream':
                    print(o.get('text', ''), end='')
                else:
                    print(f"[{otype}]", file=sys.stderr)
            print()
        print()


def cell_at(nb, idx):
    n = len(nb['cells'])
    if idx < 0 or idx >= n:
        print(f"Error: cell index {idx} out of range (0-{n-1})", file=sys.stderr)
        sys.exit(1)
    cell = nb['cells'][idx]
    cid = cell.get('id', '?')
    ct = cell['cell_type']
    print(f"Cell {idx} (id={cid}, type={ct})")
    src = ''.join(cell.get('source', []))
    if src:
        print(src, end='')
    if ct == 'code' and cell.get('outputs'):
        for o in cell['outputs']:
            otype = o.get('output_type', '?')
            if otype == 'error':
                print(f"\n[ERROR] {o.get('ename', '?')}: {o.get('evalue', '?')}", file=sys.stderr)
            elif otype in ('execute_result', 'display_data'):
                data = o.get('data', {})
                if 'text/plain' in data:
                    print(f"\n{data['text/plain']}", end='')
                elif 'text/markdown' in data:
                    print(f"\n{data['text/markdown']}", end='')
            elif otype == 'stream':
                print(o.get('text', ''), end='')
    print()


def main():
    parser = argparse.ArgumentParser(description='Inspect Jupyter notebook')
    parser.add_argument('notebook')
    parser.add_argument('--cells', action='store_true', help='Dump full cell source')
    parser.add_argument('--summary', action='store_true', help='Show summary (default)')
    parser.add_argument('--cell', type=int, help='Show specific cell')
    args = parser.parse_args()

    nb = load_nb(args.notebook)
    nb['file'] = args.notebook

    if args.cell is not None:
        cell_at(nb, args.cell)
    elif args.cells:
        full_cells(nb)
    else:
        summary(nb)


if __name__ == '__main__':
    main()
