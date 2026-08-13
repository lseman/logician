#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== TUI Latency Benchmark ==="
echo ""

# Source stats
SRC_DIR="src"
TS_FILES=$(find "$SRC_DIR" -name '*.ts' ! -path '*/__tests__/*' | wc -l)
TOTAL_LINES=0
LARGEST_FILE=""
LARGEST_SIZE=0
while IFS= read -r f; do
  lines=$(wc -l < "$f")
  TOTAL_LINES=$((TOTAL_LINES + lines))
  size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
  if [ "$size" -gt "$LARGEST_SIZE" ]; then LARGEST_SIZE=$size; LARGEST_FILE=$(basename "$f"); fi
done < <(find "$SRC_DIR" -name '*.ts' ! -path '*/__tests__/*')
echo "Source: $TS_FILES files, ~$TOTAL_LINES lines, largest: $LARGEST_FILE ($LARGEST_SIZE bytes)"

# Node cold module load (3 runs)
echo ""
echo "--- Module Load ---"
for run in 1 2 3; do
  start=$(date +%s%N)
  node --experimental-strip-types --no-warnings -e "import('./src/index.ts').catch(()=>{})" > /dev/null 2>&1 || true
  end=$(date +%s%N)
  elapsed=$(( (end - start) / 1000000 ))
  echo "Run $run: ${elapsed} ms"
done

# Render performance benchmark in Node.js (measures the hot path: cell parsing + diffing)
echo ""
echo "--- Render Performance ---"
node --experimental-strip-types --no-warnings -e '
const { performance } = await import("node:perf_hooks");

function visibleWidth(s) {
  let w = 0;
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (c === undefined) continue;
    // Skip ANSI sequences
    if (ch === "\x1b") continue;
    w += (c >= 0x1100 && (c <= 0x2FF8 || c == 0x3005 || c == 0x3007 || c >= 0x3021 && c <= 0x3029 || c >= 0xFF01 && c <= 0xFF60 || c >= 0xA13F && c <= 0xAC00 || c >= 0xD7B0 && c <= 0xD7FF)) ? 2 : 1;
  }
  return w;
}

function parseLineIntoCells(line, targetWidth) {
  const cells = [];
  let attr = "";
  let i = 0;
  const len = line.length;
  while (i < len && cells.length < targetWidth) {
    const ch = line[i];
    if (ch === "\x1b") {
      const next = line[i + 1];
      if (next === "[") { let j = i + 2; while (j < len) { const fc = line.charCodeAt(j); if (fc >= 0x40 && fc <= 0x7e) break; j++; } i = j + 1; continue; }
      if (next === "]") { let j = i + 2; while (j < len) { if (line[j] === "\x07") break; j++; } i = Math.min(len, j + 2); continue; }
      attr += ch; i++; continue;
    }
    const code = ch.charCodeAt(0);
    if (code < 0x20 && code !== 0x09) { i++; continue; }
    const cp = line.codePointAt(i);
    if (cp === undefined) break;
    const char = String.fromCodePoint(cp);
    const w = Math.max(1, char.length);
    if (w > 0 && cells.length + w <= targetWidth) { cells.push({char, attr}); }
    i += char.length;
  }
  while (cells.length < targetWidth) cells.push({char: " ", attr: ""});
  return cells;
}

function cellLevelDiff(prevCells, newCells, row) {
  const changed = new Array(newCells.length).fill(false);
  for (let i = 0; i < newCells.length; i++) {
    changed[i] = !prevCells[i] || prevCells[i].char !== newCells[i].char || prevCells[i].attr !== newCells[i].attr;
  }
  let column = 0, changedCells = 0, cursorMoves = 0;
  while (column < newCells.length) {
    if (!changed[column]) { column++; continue; }
    const start = column;
    while (column < newCells.length && changed[column]) column++;
    changedCells += column - start;
    cursorMoves++;
  }
  return { output: "", changedCells, cursorMoves };
}

// Realistic ANSI lines matching typical model output
function genLine(seed) {
  let s = ""; const c = "abcdefghijklmnopqrstuvwxyz .,";
  for (let j = 0; j < 120; j++) {
    if ((seed + j) % 3 === 0) s += "\x1b[38;5;" + (7 + (seed + j) % 90) + "m";
    s += c[Math.floor((j * 17 + seed) % c.length)];
  }
  return s + "\x1b[0m";
}

const width = 120;
// Full diff scenario (transcript streaming)
const fullLinesA = Array.from({length: 30}, (_, i) => genLine(i * 7));
const fullLinesB = fullLinesA.map(l => l.replace(/a/g, "b").replace(/c/g, "d"));

// Partial diff scenario (input bar typing - most lines unchanged)
const partialPrev = Array.from({length: 30}, (_, i) => genLine(i * 7));
const partialNew = [...partialPrev.slice(0, 29), genLine(42)]; // Only last line changed

// No change scenario (idle frame)
const idleLines = Array.from({length: 30}, (_, i) => genLine(i * 7));

function bench(label, fn) {
  const iterations = 500;
  let total = 0;
  for (let run = 0; run < iterations; run++) total += fn();
  console.log(`  ${label.padEnd(40)} ${(total / iterations).toFixed(1)} ms avg`);
}

bench("30 lines full diff", () => {
  const t0 = performance.now();
  for (let i = 0; i < fullLinesA.length; i++) cellLevelDiff(parseLineIntoCells(fullLinesA[i], width), parseLineIntoCells(fullLinesB[i], width), i);
  return performance.now() - t0;
});

bench("30 lines partial diff (1 changed)", () => {
  const t0 = performance.now();
  for (let i = 0; i < partialPrev.length; i++) cellLevelDiff(parseLineIntoCells(partialPrev[i], width), parseLineIntoCells(partialNew[i], width), i);
  return performance.now() - t0;
});

bench("30 lines no change", () => {
  const t0 = performance.now();
  for (let i = 0; i < idleLines.length; i++) cellLevelDiff(parseLineIntoCells(idleLines[i], width), parseLineIntoCells(idleLines[i], width), i);
  return performance.now() - t0;
});

' 2>&1

# tsx module load (3 runs)
echo ""
echo "--- tsx Load ---"
for run in 1 2 3; do
  start=$(date +%s%N)
  npx --yes tsx --eval "import('./src/index.ts').catch(()=>{})" > /dev/null 2>&1 || true
  end=$(date +%s%N)
  elapsed=$(( (end - start) / 1000000 ))
  echo "Run $run: ${elapsed} ms"
done

echo ""
echo "--- Summary ---"
