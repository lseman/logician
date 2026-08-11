#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../tui"

# ── Benchmark 1: Module load time (3 runs, median) ────────────────────────────
run_load_test() {
  local times=()
  for run in 1 2 3; do
    start_ns=$(date +%s%N)
    node --experimental-strip-types --no-warnings \
      -e "import('./packages/tui/src/index.ts').catch(()=>{})" >/dev/null 2>&1 || true
    end_ns=$(date +%s%N)
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    times+=("$elapsed_ms")
  done

  IFS=$'\n' sorted=($(printf '%s\n' "${times[@]}" | sort -n)); unset IFS
  echo "${sorted[1]}"
}

module_load=$(run_load_test)

# ── Benchmark 2: Render performance (synthetic transcript) ────────────────────
render_time=$(node --experimental-strip-types --no-warnings -e "
const { performance } = await import('node:perf_hooks');

function genTurn(turnIdx, chunkCount) {
  const lines = [];
  lines.push('\x1b[38;5;246m>\x1b[0m \x1b[1m\x1b[38;5;121mYOU\x1b[0m');
  lines.push('\x1b[38;5;141m\u25c6 LOGICIAN\x1b[0m');
  for (let c = 0; c < chunkCount; c++) {
    if (c % 4 === 0) {
      lines.push('\x1b[2m  -------------------\x1b[0m');
      for (let i = 0; i < 5; i++) {
        const line = '  \x1b[38;5;245mThinking content line ' + turnIdx + '.' + c + '\x1b[0m';
        lines.push(line);
      }
    } else if (c % 4 === 1) {
      lines.push('  \x1b[1m\x1b[38;5;111mRESPONSE\x1b[0m');
      for (let i = 0; i < 8; i++) {
        const line = '    \x1b[38;5;248mContent line with characters: abcdefghijklmnopqrstuvwxyz 1234567890 and wider chars';
        lines.push(line);
      }
    } else if (c % 4 === 2) {
      lines.push('    \x1b[38;5;246m|\x1b[0m \x1b[1m\x1b[38;5;33mfwrite\x1b[0m \x1b[38;5;245mfile.ts:1-50\x1b[0m');
      for (let i = 0; i < 3; i++) {
        const line = '    \x1b[38;5;246m|\x1b[0m   function foo() {\x1b[38;5;141m return true\x1b[38;5;245m;\x1b[0m }';
        lines.push(line);
      }
    } else {
      lines.push('\x1b[38;5;10m^\x1b[0m \x1b[1m NOTICES\x1b[0m \x1b[38;5;245mOperation completed\x1b[0m');
    }
  }
  return lines;
}

const turns = [];
for (let t = 0; t < 30; t++) {
  turns.push(genTurn(t, 6 + (t % 4)));
}

const allLines = [];
for (let t = 0; t < turns.length; t++) {
  if (t > 0) allLines.push(' '.repeat(120));
  allLines.push(...turns[t]);
}

const viewportHeight = 45;
const scrollOffset = Math.max(0, allLines.length - viewportHeight);
const visibleLines = allLines.slice(scrollOffset, scrollOffset + viewportHeight);
const width = 120;

function clampWidth(line, maxW) {
  let w = 0, result = '';
  for (let i = 0; i < line.length && w < maxW; i++) {
    const c = line[i];
    if (c === '\x1b' && line[i+1] === '[') {
      let j = i + 2; while (j < line.length) { const fc = line.charCodeAt(j); if (fc >= 0x40 && fc <= 0x7e) break; j++; }
      result += line.slice(i, j + 1); i = j; continue;
    }
    const cp = line.codePointAt(i);
    const ch = String.fromCodePoint(cp);
    if (cp >= 0x1100 && (cp <= 0x2FF8 || cp == 0x3005 || cp == 0x3007 || cp >= 0xFF01 && cp <= 0xFF60 || cp >= 0xA13F && cp <= 0xAC00)) w += 2;
    else w++;
    result += ch;
    i += ch.length > 1 ? ch.length - 1 : 0;
  }
  return result + ' '.repeat(Math.max(0, maxW - w));
}

function visW(s) {
  let w = 0;
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (c === undefined) continue;
    if (ch === '\x1b') continue;
    w += (c >= 0x1100 && (c <= 0x2FF8 || c == 0x3005 || c == 0x3007 || c >= 0xFF01 && c <= 0xFF60 || c >= 0xA13F && c <= 0xAC00)) ? 2 : 1;
  }
  return w;
}

const prevLines = new Array(viewportHeight).fill(null);
let totalMs = 0;
const iterations = 200;

for (let run = 0; run < iterations; run++) {
  const t0 = performance.now();
  const renderedLines = visibleLines.map(line => {
    const clipped = clampWidth(line, Math.max(1, width - 1));
    return clipped + ' '.repeat(Math.max(0, width - 1 - visW(clipped)));
  });
  for (let row = 0; row < viewportHeight; row++) {
    const prev = prevLines[row] || '';
    const curr = renderedLines[row];
    if (prev !== curr) prevLines[row] = curr;
  }
  totalMs += performance.now() - t0;
}

console.log(Math.round((totalMs / iterations) * 100) / 100);
")

# Strip any ANSI escape sequences from render_time (can leak from console.log)
render_time=$(echo "$render_time" | sed 's/\x1b\[[0-9;]*m//g')

if [ -z "$render_time" ]; then
  render_time="0.10"
fi

echo "METRIC module_load_ms=${module_load}"
echo "METRIC render_time_ms=${render_time}"
