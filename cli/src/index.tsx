import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Box, Newline, render, Text, useApp, useInput, Static } from 'ink';
import TextInput from 'ink-text-input';
import MarkdownIt = require('markdown-it/dist/index.cjs.js');
import { highlight } from 'cli-highlight';

import { PythonBridge, type BridgeEvent } from './bridge.js';

type Role = 'user' | 'assistant' | 'system' | 'trace' | 'tool' | 'skill';
type Phase = 'ready' | 'thinking' | 'bubbling' | 'jambering' | 'streaming' | 'error';

type Message = {
  id: string;
  role: Role;
  text: string;
  streaming?: boolean;
  rawStream?: string;
};

type BridgeState = {
  active: string;
  session: string;
  msg_count: number;
  agents: string[];
  pipeline: Record<string, unknown> | null;
  rapidfuzz?: boolean;
  tool_count?: number;
  skill_count?: number;
};

type SlashResult = {
  messages: string[];
  state: BridgeState;
  exit?: boolean;
};

type ChatResult = {
  pipeline: boolean;
  assistant?: string;
  turns?: Array<{ agent: string; text: string }>;
  state: BridgeState;
};

type StateResult = BridgeState;

const phaseColor: Record<Phase, string> = {
  ready: 'green',
  thinking: 'yellow',
  bubbling: 'yellowBright',
  jambering: 'magenta',
  streaming: 'cyan',
  error: 'red'
};

const now = () => new Date().toLocaleTimeString();
const mkId = () => Math.random().toString(36).slice(2, 10);
const STREAM_DRAFT_FLUSH_MS = 50;

type MdToken = ReturnType<MarkdownIt['parse']>[number];
type HorizontalAlign = 'left' | 'center' | 'right';
type ThinkSegment = {
  text: string;
  inThink: boolean;
};
type ToolCallChunk = {
  kind: 'text' | 'tool_call';
  text: string;
};

const mdParser = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: true
});

const parseAlignFromToken = (token: MdToken): HorizontalAlign => {
  const alignAttr = token.attrGet('align');
  if (alignAttr === 'center' || alignAttr === 'right' || alignAttr === 'left') {
    return alignAttr;
  }

  const style = token.attrGet('style') || '';
  const m = style.match(/text-align\s*:\s*(left|center|right)/i);
  if (m?.[1] === 'center' || m?.[1] === 'right' || m?.[1] === 'left') {
    return m[1];
  }

  return 'left';
};

const ruleLine = (indent = 0): string => {
  const cols = process.stdout.columns ?? 100;
  const width = Math.max(16, Math.min(72, cols - 10 - indent));
  return `${' '.repeat(indent)}${'─'.repeat(width)}`;
};

const truncate = (value: string, max: number): string => {
  if (value.length <= max) {
    return value;
  }
  if (max <= 3) {
    return '.'.repeat(Math.max(1, max));
  }
  return `${value.slice(0, max - 3)}...`;
};

const findMatching = (
  tokens: MdToken[],
  start: number,
  openType: string,
  closeType: string
): number => {
  let depth = 0;
  for (let i = start; i < tokens.length; i += 1) {
    if (tokens[i].type === openType) {
      depth += 1;
    } else if (tokens[i].type === closeType) {
      depth -= 1;
      if (depth === 0) {
        return i;
      }
    }
  }
  return start;
};

const toPlainInline = (token?: MdToken): string => {
  if (!token) {
    return '';
  }
  if (token.type === 'inline') {
    return (token.children || [])
      .map((t: MdToken) => {
        if (t.type === 'text' || t.type === 'code_inline' || t.type === 'html_inline') {
          return t.content;
        }
        if (t.type === 'image') {
          const src = t.attrGet('src') || '';
          return `[image:${t.content || 'unnamed'}](${src})`;
        }
        if (t.type === 'softbreak' || t.type === 'hardbreak') {
          return ' ';
        }
        return t.content || '';
      })
      .join('')
      .trim();
  }
  return (token.content || '').trim();
};

const isLineBreakChar = (value: string | null | undefined): boolean => {
  return value === '\n' || value === '\r';
};

const findBalancedObjectEnd = (text: string, start: number): number => {
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (ch === '\\') {
        escaped = true;
        continue;
      }
      if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === '{') {
      depth += 1;
      continue;
    }
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        return i;
      }
      continue;
    }
  }
  return -1;
};

const extractToolCall = (value: unknown): { name: string; args: unknown } | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const record = value as Record<string, unknown>;
  const toolCall = record.tool_call;
  if (!toolCall || typeof toolCall !== 'object') {
    return null;
  }
  const callRecord = toolCall as Record<string, unknown>;
  const name = typeof callRecord.name === 'string' ? callRecord.name.trim() : '';
  if (!name) {
    return null;
  }
  return { name, args: callRecord.arguments ?? {} };
};

const summarizeToolArgs = (args: unknown): string => {
  try {
    const raw = JSON.stringify(args ?? {});
    return truncate(raw || '{}', 200);
  } catch {
    return '{}';
  }
};

const splitToolCallChunks = (text: string): ToolCallChunk[] => {
  if (!text.includes('tool_call')) {
    return [{ kind: 'text', text }];
  }

  const chunks: ToolCallChunk[] = [];
  let cursor = 0;

  for (let i = 0; i < text.length; i += 1) {
    if (text[i] !== '{') {
      continue;
    }
    const end = findBalancedObjectEnd(text, i);
    if (end < 0) {
      continue;
    }
    const raw = text.slice(i, end + 1);
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      try {
        parsed = JSON.parse(raw.replace(/[“”]/g, '"').replace(/[‘’]/g, "'"));
      } catch {
        continue;
      }
    }

    const call = extractToolCall(parsed);
    if (!call) {
      continue;
    }

    if (i > cursor) {
      chunks.push({ kind: 'text', text: text.slice(cursor, i) });
    }

    const before = i > 0 ? text[i - 1] : '';
    const after = end + 1 < text.length ? text[end + 1] : '';
    const args = summarizeToolArgs(call.args);
    const leading = before && !isLineBreakChar(before) ? '\n' : '';
    const trailing = after && !isLineBreakChar(after) ? '\n' : '';
    const summary = `${leading}calling tool: ${call.name}\nparameters: ${args}${trailing}`;
    chunks.push({ kind: 'tool_call', text: summary });

    cursor = end + 1;
    i = end;
  }

  if (!chunks.length) {
    return [{ kind: 'text', text }];
  }
  if (cursor < text.length) {
    chunks.push({ kind: 'text', text: text.slice(cursor) });
  }
  return chunks;
};

const splitThinkSegments = (
  text: string,
  initialDepth: number,
  previousChar: string
): { segments: ThinkSegment[]; nextDepth: number; nextChar: string } => {
  const segments: ThinkSegment[] = [];
  let depth = initialDepth;
  let last = 0;
  let tailChar = previousChar;
  const tagRe = /<\/?\s*think\s*>/gi;

  for (const match of text.matchAll(tagRe)) {
    const idx = match.index ?? 0;
    if (idx > last) {
      const segmentText = text.slice(last, idx);
      segments.push({ text: segmentText, inThink: depth > 0 });
      tailChar = segmentText[segmentText.length - 1] || tailChar;
    }
    const raw = match[0] || '';
    const isClose = /^<\s*\/\s*think\s*>$/i.test(raw);
    if (isClose) {
      depth = Math.max(0, depth - 1);
    } else {
      if (depth === 0 && !isLineBreakChar(tailChar)) {
        segments.push({ text: '\n', inThink: false });
        tailChar = '\n';
      }
      depth += 1;
    }
    last = idx + raw.length;
  }

  if (last < text.length) {
    const segmentText = text.slice(last);
    segments.push({ text: segmentText, inThink: depth > 0 });
    tailChar = segmentText[segmentText.length - 1] || tailChar;
  }

  return { segments, nextDepth: depth, nextChar: tailChar };
};

const renderInlineTokens = (
  tokens?: MdToken[] | null,
  keyPrefix = 'in'
): React.ReactNode[] => {
  if (!tokens?.length) {
    return [];
  }

  const out: React.ReactNode[] = [];
  let seq = 0;
  let bold = 0;
  let italic = 0;
  let strike = 0;
  const linkStack: string[] = [];
  let thinkDepth = 0;
  let lastChar = '\n';

  const pushText = (text: string, color?: string, code = false) => {
    if (!text) {
      return;
    }
    const { segments, nextDepth, nextChar } = splitThinkSegments(text, thinkDepth, lastChar);
    thinkDepth = nextDepth;
    lastChar = nextChar;
    for (const segment of segments) {
      if (!segment.text) {
        continue;
      }
      const toolChunks = code ? [{ kind: 'text', text: segment.text } satisfies ToolCallChunk] : splitToolCallChunks(segment.text);
      for (const chunk of toolChunks) {
        if (!chunk.text) {
          continue;
        }
        const isToolChunk = chunk.kind === 'tool_call';
        out.push(
          <Text
            key={`${keyPrefix}-${seq++}`}
            bold={bold > 0 || isToolChunk}
            italic={italic > 0 || segment.inThink}
            strikethrough={strike > 0}
            underline={linkStack.length > 0 && !isToolChunk}
            color={
              isToolChunk
                ? 'cyanBright'
                : segment.inThink
                  ? 'magentaBright'
                  : color || (linkStack.length > 0 ? 'blueBright' : 'white')
            }
          >
            {chunk.text}
          </Text>
        );
      }
    }
  };

  for (const token of tokens) {
    switch (token.type) {
      case 'text':
      case 'html_inline':
        pushText(token.content || '');
        break;
      case 'code_inline':
        pushText(token.content || '', 'cyanBright', true);
        break;
      case 'softbreak':
      case 'hardbreak':
        pushText('\n');
        break;
      case 'strong_open':
        bold += 1;
        break;
      case 'strong_close':
        bold = Math.max(0, bold - 1);
        break;
      case 'em_open':
        italic += 1;
        break;
      case 'em_close':
        italic = Math.max(0, italic - 1);
        break;
      case 's_open':
        strike += 1;
        break;
      case 's_close':
        strike = Math.max(0, strike - 1);
        break;
      case 'link_open': {
        linkStack.push(token.attrGet('href') || '');
        break;
      }
      case 'link_close': {
        const href = linkStack.pop();
        if (href) {
          pushText(` (${href})`, 'blue');
        }
        break;
      }
      case 'image': {
        const src = token.attrGet('src') || '';
        pushText(`[image: ${token.content || 'unnamed'}]`, 'magentaBright');
        if (src) {
          pushText(` (${src})`, 'blue');
        }
        break;
      }
      default:
        if (token.content) {
          pushText(token.content);
        }
        break;
    }
  }

  return out;
};

const renderTable = (
  header: string[],
  aligns: HorizontalAlign[],
  rows: string[][],
  key: string
): React.ReactNode => {
  const cols = Math.max(header.length, ...rows.map(r => r.length));
  const normalizedHeader = Array.from({ length: cols }, (_, i) => header[i] || '');
  const normalizedRows = rows.map(row => Array.from({ length: cols }, (_, i) => row[i] || ''));
  const allRows = [normalizedHeader, ...normalizedRows];

  const colWidths = Array.from({ length: cols }, (_, c) =>
    Math.max(3, ...allRows.map(r => (r[c] || '').length))
  );
  const minWidths = Array.from({ length: cols }, () => 3);

  const calcTableWidth = () =>
    1 + colWidths.reduce((sum, width) => sum + width + 2, 0) + Math.max(0, cols - 1) + 1;

  const maxWidth = Math.max(40, (process.stdout.columns ?? 120) - 6);
  while (calcTableWidth() > maxWidth) {
    let idx = -1;
    let best = -1;
    for (let i = 0; i < colWidths.length; i += 1) {
      const slack = colWidths[i] - minWidths[i];
      if (slack > best) {
        best = slack;
        idx = i;
      }
    }
    if (idx < 0 || best <= 0) {
      break;
    }
    colWidths[idx] -= 1;
  }

  const padCell = (value: string, width: number, align: HorizontalAlign): string => {
    const cell = truncate(value, width);
    if (cell.length >= width) {
      return cell;
    }
    const remain = width - cell.length;
    if (align === 'right') {
      return `${' '.repeat(remain)}${cell}`;
    }
    if (align === 'center') {
      const left = Math.floor(remain / 2);
      return `${' '.repeat(left)}${cell}${' '.repeat(remain - left)}`;
    }
    return `${cell}${' '.repeat(remain)}`;
  };

  const rowText = (cells: string[]) =>
    cells
      .map((cell, i) => ` ${padCell(cell || '', colWidths[i], aligns[i] || 'left')} `)
      .join('│');

  const top = `┌${colWidths.map(w => '─'.repeat(w + 2)).join('┬')}┐`;
  const mid = `├${colWidths.map(w => '─'.repeat(w + 2)).join('┼')}┤`;
  const bot = `└${colWidths.map(w => '─'.repeat(w + 2)).join('┴')}┘`;

  return (
    <Box key={key} flexDirection="column">
      <Text color="gray">{top}</Text>
      <Text color="yellowBright">{`│${rowText(normalizedHeader)}│`}</Text>
      <Text color="gray">{mid}</Text>
      {normalizedRows.map((row, i) => (
        <Text key={`${key}-row-${i}`} color="white">
          {`│${rowText(row)}│`}
        </Text>
      ))}
      <Text color="gray">{bot}</Text>
    </Box>
  );
};

const renderBlocks = (
  tokens: MdToken[],
  start: number,
  end: number,
  keyPrefix: string,
  indent = 0
): React.ReactNode[] => {
  const rendered: React.ReactNode[] = [];
  let i = start;
  let seq = 0;

  while (i < end) {
    const token = tokens[i];
    const key = `${keyPrefix}-${seq++}`;

    if (token.type === 'heading_open') {
      const inline = tokens[i + 1];
      const tag = token.tag;
      const color: Record<string, string> = {
        h1: 'yellowBright',
        h2: 'greenBright',
        h3: 'cyanBright',
        h4: 'blueBright',
        h5: 'magentaBright',
        h6: 'gray'
      };
      rendered.push(
        <Text key={key} color={color[tag] || 'white'} bold>
          {' '.repeat(indent)}
          {renderInlineTokens(inline?.children, `${key}-h`)}
        </Text>
      );
      i += 3;
      continue;
    }

    if (token.type === 'paragraph_open') {
      const close = findMatching(tokens, i, 'paragraph_open', 'paragraph_close');
      const inline = tokens[i + 1];
      rendered.push(
        <Text key={key} color="white">
          {' '.repeat(indent)}
          {renderInlineTokens(inline?.children, `${key}-p`)}
        </Text>
      );
      i = close + 1;
      continue;
    }

    if (token.type === 'inline') {
      rendered.push(
        <Text key={key} color="white">
          {' '.repeat(indent)}
          {renderInlineTokens(token.children, `${key}-in`) || token.content}
        </Text>
      );
      i += 1;
      continue;
    }

    if (token.type === 'fence' || token.type === 'code_block') {
      const lang = (token.info || '').trim().split(/\s+/)[0] || '';
      const rawCode = (token.content || '').replace(/\n$/, '');
      let highlighted = rawCode;
      try {
        highlighted = highlight(rawCode, { language: lang || 'txt', ignoreIllegals: true });
      } catch {
        // Fallback
      }
      rendered.push(
        <Box key={key} flexDirection="column" marginLeft={indent}>
          {lang ? <Text color="gray">{`code:${lang}`}</Text> : null}
          <Box borderStyle="round" borderColor="gray" paddingX={1} flexDirection="column">
            <Text>{highlighted || ' '}</Text>
          </Box>
        </Box>
      );
      i += 1;
      continue;
    }

    if (token.type === 'blockquote_open') {
      const close = findMatching(tokens, i, 'blockquote_open', 'blockquote_close');
      rendered.push(
        <Box key={key} borderStyle="round" borderColor="cyan" paddingX={1} flexDirection="column" marginLeft={1}>
          {renderBlocks(tokens, i + 1, close, `${key}-q`, indent + 1)}
        </Box>
      );
      i = close + 1;
      continue;
    }

    if (token.type === 'bullet_list_open' || token.type === 'ordered_list_open') {
      const ordered = token.type === 'ordered_list_open';
      const closeType = ordered ? 'ordered_list_close' : 'bullet_list_close';
      const close = findMatching(tokens, i, token.type, closeType);
      const startAt = ordered ? Number(token.attrGet('start') || '1') : 1;
      const items: React.ReactNode[] = [];
      let j = i + 1;
      let itemIdx = 0;

      while (j < close) {
        if (tokens[j].type === 'list_item_open') {
          const itemClose = findMatching(tokens, j, 'list_item_open', 'list_item_close');
          const marker = ordered ? `${startAt + itemIdx}.` : '•';
          items.push(
            <Box key={`${key}-li-${itemIdx}`} flexDirection="row">
              <Text color="yellow">{`${' '.repeat(indent)}${marker} `}</Text>
              <Box flexDirection="column">{renderBlocks(tokens, j + 1, itemClose, `${key}-item-${itemIdx}`, indent)}</Box>
            </Box>
          );
          itemIdx += 1;
          j = itemClose + 1;
          continue;
        }
        j += 1;
      }

      rendered.push(
        <Box key={key} flexDirection="column">
          {items}
        </Box>
      );
      i = close + 1;
      continue;
    }

    if (token.type === 'table_open') {
      const close = findMatching(tokens, i, 'table_open', 'table_close');
      let inHeader = false;
      let currentRow: string[] = [];
      let currentAligns: HorizontalAlign[] = [];
      let header: string[] = [];
      let aligns: HorizontalAlign[] = [];
      const rows: string[][] = [];

      for (let j = i + 1; j < close; j += 1) {
        const t = tokens[j];
        if (t.type === 'thead_open') {
          inHeader = true;
          continue;
        }
        if (t.type === 'thead_close') {
          inHeader = false;
          continue;
        }
        if (t.type === 'tr_open') {
          currentRow = [];
          currentAligns = [];
          continue;
        }
        if (t.type === 'th_open' || t.type === 'td_open') {
          const closeCell = findMatching(
            tokens,
            j,
            t.type,
            t.type === 'th_open' ? 'th_close' : 'td_close'
          );
          currentRow.push(toPlainInline(tokens[j + 1]));
          if (t.type === 'th_open') {
            currentAligns.push(parseAlignFromToken(t));
          }
          j = closeCell;
          continue;
        }
        if (t.type === 'tr_close') {
          if (inHeader && !header.length) {
            header = currentRow;
            aligns = currentAligns;
          } else if (currentRow.length) {
            rows.push(currentRow);
          }
        }
      }

      if (!header.length && rows.length) {
        header = rows.shift() || [];
      }
      if (!aligns.length) {
        aligns = Array.from({ length: header.length }, () => 'left');
      }

      rendered.push(renderTable(header, aligns, rows, key));
      i = close + 1;
      continue;
    }

    if (token.type === 'hr') {
      rendered.push(
        <Text key={key} color="gray">
          {ruleLine(indent)}
        </Text>
      );
      i += 1;
      continue;
    }

    if (token.type === 'html_block') {
      const html = (token.content || '').trim();
      if (html) {
        rendered.push(
          <Text key={key} color="gray">
            {`${' '.repeat(indent)}${html}`}
          </Text>
        );
      }
      i += 1;
      continue;
    }

    i += 1;
  }

  return rendered;
};

const MarkdownBlockBase: React.FC<{ content: string }> = ({ content }) => {
  const tokens = useMemo(() => mdParser.parse(content, {}), [content]);
  const rendered = useMemo(
    () => renderBlocks(tokens, 0, tokens.length, 'md', 0),
    [tokens]
  );

  return <Box flexDirection="column">{rendered}</Box>;
};
const MarkdownBlock = React.memo(MarkdownBlockBase);
MarkdownBlock.displayName = 'MarkdownBlock';

const renderLiveStreamContent = (content: string, keyPrefix: string): React.ReactNode[] => {
  if (!content) {
    return [];
  }
  const out: React.ReactNode[] = [];
  let seq = 0;
  const { segments } = splitThinkSegments(content, 0, '\n');
  for (const segment of segments) {
    if (!segment.text) {
      continue;
    }
    const toolChunks = splitToolCallChunks(segment.text);
    for (const chunk of toolChunks) {
      if (!chunk.text) {
        continue;
      }
      const isTool = chunk.kind === 'tool_call';
      out.push(
        <Text
          key={`${keyPrefix}-${seq++}`}
          color={isTool ? 'cyanBright' : segment.inThink ? 'magentaBright' : 'white'}
          bold={isTool}
          italic={segment.inThink}
        >
          {chunk.text}
        </Text>
      );
    }
  }
  return out;
};

const LiveStreamBlockBase: React.FC<{ content: string }> = ({ content }) => {
  const rendered = useMemo(() => renderLiveStreamContent(content, 'live'), [content]);
  if (!rendered.length) {
    return null;
  }
  return <>{rendered}</>;
};
const LiveStreamBlock = React.memo(LiveStreamBlockBase);
LiveStreamBlock.displayName = 'LiveStreamBlock';

const RoleTag: React.FC<{ role: Role }> = ({ role }) => {
  if (role === 'user') {
    return <Text color="yellow">you</Text>;
  }
  if (role === 'assistant') {
    return <Text color="green">assistant</Text>;
  }
  if (role === 'trace') {
    return <Text color="gray" dimColor>trace</Text>;
  }
  if (role === 'tool') {
    return <Text color="cyanBright">tool ⚙</Text>;
  }
  if (role === 'skill') {
    return <Text color="magentaBright">skill 🧠</Text>;
  }
  return <Text color="gray">system</Text>;
};

const MessageRow: React.FC<{ message: Message; rawOn: boolean }> = React.memo(({ message, rawOn }) => {
  if (message.role === 'assistant' || message.role === 'system') {
    const isAssistant = message.role === 'assistant';
    const isLiveAssistant = isAssistant && Boolean(message.streaming);
    const showRawAssistant = isAssistant && !isLiveAssistant && rawOn && Boolean(message.rawStream);
    return (
      <Box flexDirection="row" marginBottom={1} alignItems="flex-start">
        <Box width={12}>
          <RoleTag role={message.role} />
        </Box>
        <Box flexDirection="column" flexGrow={1}>
          {isLiveAssistant ? (
            <LiveStreamBlock content={message.text} />
          ) : showRawAssistant ? (
            <LiveStreamBlock content={message.rawStream || ''} />
          ) : (
            <MarkdownBlock content={message.text} />
          )}
        </Box>
      </Box>
    );
  }

  if (message.role === 'trace') {
    return (
      <Box marginBottom={1}>
        <Box width={12}>
          <RoleTag role={message.role} />
        </Box>
        <Text color="gray" dimColor>
          {message.text}
        </Text>
      </Box>
    );
  }

  if (message.role === 'tool') {
    const lines = message.text.split('\n');
    const name = lines[0];
    const argsJson = lines.slice(1).join('\n');
    let highlightedArgs = argsJson;
    try {
      highlightedArgs = highlight(argsJson, { language: 'json', ignoreIllegals: true });
    } catch {
      // Ignore
    }
    return (
      <Box marginBottom={1} flexDirection="row" alignItems="flex-start" width="100%">
        <Box width={12} flexShrink={0}>
          <RoleTag role={message.role} />
        </Box>
        <Box flexDirection="column" borderStyle="round" borderColor="cyan" paddingX={1} flexGrow={1} flexShrink={1}>
          <Text color="cyanBright" bold>{name}</Text>
          <Text>{highlightedArgs}</Text>
        </Box>
      </Box>
    );
  }

  if (message.role === 'skill') {
    return (
      <Box marginBottom={1} flexDirection="row" alignItems="flex-start" width="100%">
        <Box width={12} flexShrink={0}>
          <RoleTag role={message.role} />
        </Box>
        <Box flexDirection="column" borderStyle="round" borderColor="magenta" paddingX={1} flexGrow={1} flexShrink={1}>
          <Text color="magentaBright">{message.text}</Text>
        </Box>
      </Box>
    );
  }

  return (
    <Box marginBottom={1}>
      <Box width={12}>
        <RoleTag role={message.role} />
      </Box>
      <Text>{message.text}</Text>
    </Box>
  );
});
MessageRow.displayName = 'MessageRow';



const PromptInput: React.FC<{
  onSubmit: (value: string) => Promise<void>;
}> = React.memo(({ onSubmit }) => {
  const [value, setValue] = useState('');

  const handleSubmit = useCallback((raw: string) => {
    setValue('');
    void onSubmit(raw);
  }, [onSubmit]);

  return (
    <Box>
      <Text color="yellow">❯ </Text>
      <TextInput
        value={value}
        onChange={setValue}
        onSubmit={handleSubmit}
        placeholder="type message or /command"
      />
    </Box>
  );
});
PromptInput.displayName = 'PromptInput';

const coercePhase = (value: string): Phase => {
  const v = value.toLowerCase();
  if (v === 'thinking') return 'thinking';
  if (v === 'bubbling' || v === 'bubblering') return 'bubbling';
  if (v === 'jambering' || v === 'hjambering') return 'jambering';
  if (v === 'streaming') return 'streaming';
  if (v === 'error') return 'error';
  return 'ready';
};

const App: React.FC = () => {
  const { exit } = useApp();

  const [messages, setMessages] = useState<Message[]>([]);
  const [traceOn, setTraceOn] = useState(true);
  const [streamPanelOn, setStreamPanelOn] = useState(false);
  const [phase, setPhase] = useState<Phase>('ready');
  const [phaseNote, setPhaseNote] = useState('ready');
  const [busy, setBusy] = useState(false);
  const [spinnerTick, setSpinnerTick] = useState(0);
  const [state, setState] = useState<BridgeState>({
    active: '-',
    session: '-',
    msg_count: 0,
    agents: [],
    pipeline: null,
    rapidfuzz: false,
    tool_count: 0,
    skill_count: 0,
  });

  const bridgeRef = useRef<PythonBridge | null>(null);
  const traceOnRef = useRef(false);
  const busyRef = useRef(false);
  const traceEventsRef = useRef<string[]>([]);
  const draftBufferRef = useRef('');
  const draftFlushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const streamRawRef = useRef('');
  const phaseCycleRef = useRef<Phase[]>(['thinking', 'bubbling', 'jambering']);
  const phaseIdxRef = useRef(0);
  const submitHandlerRef = useRef<(value: string) => Promise<void>>(async () => { });
  const liveAssistantIdRef = useRef<string | null>(null);

  const pushTrace = (event: string) => {
    const line = `${now()}  ${event}`;
    const next = [...traceEventsRef.current.slice(-159), line];
    traceEventsRef.current = next;
  };

  const upsertLiveAssistantChunk = (chunk: string) => {
    if (!chunk) {
      return;
    }

    setMessages(prev => {
      const liveId = liveAssistantIdRef.current;

      if (!liveId) {
        const id = mkId();
        liveAssistantIdRef.current = id;
        return [...prev, { id, role: 'assistant', text: chunk, streaming: true }];
      }

      const index = prev.findIndex(message => message.id === liveId);
      if (index < 0) {
        const id = mkId();
        liveAssistantIdRef.current = id;
        return [...prev, { id, role: 'assistant', text: chunk, streaming: true }];
      }

      const current = prev[index];
      const next = [...prev];
      next[index] = {
        ...current,
        role: 'assistant',
        text: `${current.text}${chunk}`,
        streaming: true
      };
      return next;
    });
  };

  const flushDraftBuffer = () => {
    if (!draftBufferRef.current) {
      return;
    }
    const chunk = draftBufferRef.current;
    draftBufferRef.current = '';
    upsertLiveAssistantChunk(chunk);
  };

  const scheduleDraftFlush = () => {
    if (draftFlushTimerRef.current) {
      return;
    }
    draftFlushTimerRef.current = setTimeout(() => {
      draftFlushTimerRef.current = null;
      flushDraftBuffer();
    }, STREAM_DRAFT_FLUSH_MS);
  };

  const flushDraftNow = () => {
    if (draftFlushTimerRef.current) {
      clearTimeout(draftFlushTimerRef.current);
      draftFlushTimerRef.current = null;
    }
    flushDraftBuffer();
  };

  const addMessage = (
    role: Role,
    text: string,
    extras?: Pick<Message, 'streaming' | 'rawStream'>
  ) => {
    setMessages(prev => [...prev, { id: mkId(), role, text, ...(extras || {}) }]);
  };

  const finalizeLiveAssistantMessage = (finalText: string) => {
    flushDraftNow();
    const liveId = liveAssistantIdRef.current;
    const rawStream = streamRawRef.current;
    liveAssistantIdRef.current = null;

    // Normalize: strip the payload the bridge sends at end — it replicates the
    // full text the model just streamed and causes the duplicate '?' ghost message.
    const trimmedFinal = (finalText || '').trim();

    setMessages(prev => {
      const index = liveId != null ? prev.findIndex(message => message.id === liveId) : -1;

      if (index >= 0) {
        // A live streamed message exists — ALWAYS keep its text exactly as received.
        // The bridge's final payload is a re-assembly of all streamed tokens and
        // must never replace or trim the streamed content.
        const current = prev[index];
        const next = [...prev];
        next[index] = {
          ...current,
          role: 'assistant' as const,
          text: current.text,  // unconditionally keep the streamed text
          streaming: false,
          rawStream: rawStream || current.rawStream
        };
        return next;
      }

      // No live streamed message at all — only create a new one when the
      // final payload is a meaningful string (not just punctuation/whitespace).
      const isMeaningful = trimmedFinal.length > 2 && /[a-zA-Z0-9]/.test(trimmedFinal);
      if (!isMeaningful) {
        return prev;
      }
      return [
        ...prev,
        {
          id: mkId(),
          role: 'assistant' as const,
          text: trimmedFinal,
          rawStream: rawStream || undefined
        }
      ];
    });
  };

  const stopLiveAssistantMessage = () => {
    flushDraftNow();
    const liveId = liveAssistantIdRef.current;
    liveAssistantIdRef.current = null;
    if (!liveId) {
      return;
    }

    setMessages(prev => {
      const index = prev.findIndex(message => message.id === liveId);
      if (index < 0) {
        return prev;
      }

      const current = prev[index];
      const next = [...prev];
      next[index] = {
        ...current,
        streaming: false,
        rawStream: streamRawRef.current || current.rawStream
      };
      return next;
    });
  };

  const addTraceMessage = (text: string) => {
    setMessages(prev => [...prev, { id: mkId(), role: 'trace', text }]);
  };

  const formatStatus = (s: BridgeState): string => {
    const agents = s.agents.length ? s.agents.join(', ') : '-';
    const sessionShort = s.session ? s.session.slice(0, 24) : '-';
    const pipeline = s.pipeline
      ? `${String((s.pipeline as Record<string, unknown>).a || '?')} -> ${String((s.pipeline as Record<string, unknown>).b || '?')} x${String((s.pipeline as Record<string, unknown>).rounds || '?')}`
      : 'off';
    return [
      `active: ${s.active || '-'}`,
      `session: ${sessionShort}`,
      `messages: ${s.msg_count}`,
      `agents: ${agents}`,
      `pipeline: ${pipeline}`,
      `rapidfuzz: ${s.rapidfuzz ? 'enabled' : 'disabled'}`,
      `trace: ${traceOnRef.current ? 'on' : 'off'}`
    ].join('\n');
  };

  const formatHelp = (): string => {
    return [
      '## Commands',
      '',
      '| Command | What it does |',
      '| --- | --- |',
      '| `/help` | Show this command list |',
      '| `/status` | Show runtime state snapshot |',
      '| `/doctor` | Run local health checks |',
      '| `/bug [note]` | Save a reproducible bug report file |',
      '| `/trace [on|off]` | Toggle trace messages in transcript |',
      '| `/clear` | Clear visible transcript only |',
      '| `/agents` | List loaded agents |',
      '| `/agent <name>` | Switch active agent |',
      '| `/pipeline <a> <b> [rounds]` | Enable inter-agent pipeline |',
      '| `/pipeline stop` | Disable current pipeline |',
      '| `/context` | Show session/data context |',
      '| `/sessions` / `/load <id>` | List and load previous sessions |',
      '| `/export [path]` | Export chat history to markdown |',
      '| `/upload <file> [label]` | Ingest one document into RAG |',
      '| `/upload-dir <dir> [glob] [max]` | Bulk ingest documents into RAG |',
      '| `/new` | Start a new session |',
      '| `/reload` | Reload config and agents |',
      '| `/quit` | Exit CLI |',
      '',
      'Shortcuts: `Ctrl+O` trace toggle, `Ctrl+P` raw stream inline toggle, `Ctrl+C` exit.'
    ].join('\n');
  };

  const runDoctor = async () => {
    const bridge = bridgeRef.current;
    const checks: string[] = [];
    checks.push(`# Doctor`);
    checks.push('');
    checks.push(`- cwd: \`${process.cwd()}\``);
    checks.push(`- terminal columns: ${process.stdout.columns ?? 0}`);
    checks.push(`- trace: ${traceOnRef.current ? 'on' : 'off'}`);
    checks.push(`- busy: ${busyRef.current ? 'yes' : 'no'}`);
    checks.push(`- bridge ref: ${bridge ? 'present' : 'missing'}`);

    if (!bridge) {
      checks.push('- bridge state call: failed (bridge unavailable)');
      addMessage('system', checks.join('\n'));
      return;
    }

    try {
      const snap = await bridge.call<StateResult>('state', {});
      setState(snap);
      checks.push('- bridge state call: ok');
      checks.push(`- active agent: ${snap.active || '-'}`);
      checks.push(`- session: ${snap.session || '-'}`);
      checks.push(`- persisted messages: ${snap.msg_count}`);
      checks.push(`- agents loaded: ${snap.agents.length}`);
      checks.push(`- pipeline: ${snap.pipeline ? 'on' : 'off'}`);
      checks.push(`- rapidfuzz: ${snap.rapidfuzz ? 'enabled' : 'disabled'}`);
      addMessage('system', checks.join('\n'));
    } catch (error) {
      checks.push(`- bridge state call: failed (${(error as Error).message})`);
      addMessage('system', checks.join('\n'));
    }
  };

  const runBugReport = async (raw: string) => {
    const bridge = bridgeRef.current;
    const note = raw.replace(/^\/bug\s*/i, '').trim();
    let snap: BridgeState | null = null;

    if (bridge) {
      try {
        snap = await bridge.call<StateResult>('state', {});
        setState(snap);
      } catch {
        snap = null;
      }
    }

    const ts = new Date().toISOString();
    const safeStamp = ts.replace(/[:.]/g, '-');
    const reportDir = path.join(process.cwd(), 'bug_reports');
    const reportPath = path.join(reportDir, `bug_${safeStamp}.md`);
    const recentMessages = messages.slice(-25);
    const recentTrace = traceEventsRef.current.slice(-60);
    const liveDraftId = liveAssistantIdRef.current;
    const liveDraft = liveDraftId
      ? messages.find(message => message.id === liveDraftId)?.text || ''
      : '';

    const body = [
      '# CLI Bug Report',
      '',
      `- created_at: ${ts}`,
      `- cwd: ${process.cwd()}`,
      `- trace_on: ${traceOnRef.current}`,
      `- phase: ${phase}`,
      `- phase_note: ${phaseNote}`,
      `- terminal_cols: ${process.stdout.columns ?? 0}`,
      `- terminal_rows: ${process.stdout.rows ?? 0}`,
      `- note: ${note || '(none)'}`,
      `- state: ${snap ? JSON.stringify(snap) : 'unavailable'}`,
      '',
      '## Recent Trace',
      '',
      '```text',
      ...(recentTrace.length ? recentTrace : ['(none)']),
      '```',
      '',
      '## Recent Messages',
      '',
      '```text',
      ...recentMessages.map(m => `[${m.role}] ${m.text}`),
      ...(liveDraft ? [`[assistant_draft] ${liveDraft}`] : []),
      '```',
      ''
    ].join('\n');

    await mkdir(reportDir, { recursive: true });
    await writeFile(reportPath, body, 'utf-8');
    addMessage('system', `Bug report written: ${reportPath}`);
    pushTrace(`bug_report=${reportPath}`);
  };

  useEffect(() => {
    traceOnRef.current = traceOn;
  }, [traceOn]);

  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  const setIdle = () => {
    flushDraftNow();
    busyRef.current = false;
    setBusy(false);
    setPhase('ready');
    setPhaseNote('ready');
    phaseIdxRef.current = 0;
  };

  useEffect(() => {
    const bridge = new PythonBridge((event: BridgeEvent) => {
      if (event.event === 'token' && typeof event.token === 'string') {
        draftBufferRef.current += event.token;
        streamRawRef.current += event.token;
        scheduleDraftFlush();
        if (!busyRef.current) {
          busyRef.current = true;
          setBusy(true);
        }
        setPhase('streaming');
        setPhaseNote('streaming');
        return;
      }

      // Any other event (tool, skill, phase) means the text stream is interrupted.
      // We must close the current live assistant message, so if the assistant
      // streams more tokens later, it creates a NEW message below these events!
      stopLiveAssistantMessage();

      if (event.event === 'phase') {
        const stateValue = String(event.state || 'ready');
        const note = String(event.note || stateValue);
        setPhase(coercePhase(stateValue));
        setPhaseNote(note);
        pushTrace(`phase=${stateValue} note=${note}`);
        if (traceOnRef.current && (stateValue !== 'streaming' || note !== 'streaming')) {
          addTraceMessage(`phase: ${stateValue} (${note})`);
        }
        return;
      }

      if (event.event === 'tool') {
        const toolName = String(event.name || 'unknown');
        const argsPreview = JSON.stringify(event.args || {}).slice(0, 220);
        const argsFull = JSON.stringify(event.args || {}, null, 2);
        pushTrace(`tool=${toolName} args=${argsPreview}`);
        addMessage('tool', `${toolName}\n${argsFull}`);
        return;
      }

      if (event.event === 'skill') {
        const skills = Array.isArray(event.skill_ids) && event.skill_ids.length
          ? event.skill_ids.join(', ')
          : 'none';
        const tools = Array.isArray(event.selected_tools) && event.selected_tools.length
          ? event.selected_tools.join(', ')
          : 'none';
        pushTrace(`skills=${skills} selected_tools=${tools}`);
        addMessage('skill', `Activated skills: ${skills}\nAvailable tools: ${tools}`);
        return;
      }

      if (event.event === 'bridge_stderr') {
        const rawStderr = String((event as { text?: string }).text || '');
        const stderrText = rawStderr.replace(/\s+/g, ' ').trim().slice(0, 220);
        pushTrace(`stderr=${stderrText}`);
        if (traceOnRef.current && stderrText) {
          addTraceMessage(`stderr: ${stderrText}`);
        }
        return;
      }

      if (event.event === 'bridge_exit') {
        setPhase('error');
        setPhaseNote('bridge exited');
        const code = String((event as { code?: number }).code ?? '');
        pushTrace(`bridge_exit code=${code}`);
        if (traceOnRef.current) {
          addTraceMessage(`bridge exit: code=${code || 'unknown'}`);
        }
        return;
      }

      pushTrace(`event=${event.event}`);
    });

    bridge.start();
    bridgeRef.current = bridge;

    (async () => {
      try {
        const init = await bridge.call<{ config_path: string; state: BridgeState }>('init', {
          config_path: 'agent_config.json'
        });
        setState(init.state);
        // Use cyan ANSI code for the header, resetting afterward.
        // Ink's <Text> component natively supports rendering these colors.
        const header = '\x1b[36m# Logician CLI\x1b[0m';
        addMessage(
          'system',
          `${header}\n\n**Agents**: ${init.state.agents.join(', ') || '-'}\n**Rapidfuzz**: ${init.state.rapidfuzz ? 'enabled' : 'disabled'}\n\nActive agent: ${init.state.active}. Session: ${init.state.session}. Raw stream is on by default (Ctrl+P toggles).`
        );
        pushTrace(`init config=${init.config_path}`);
      } catch (error) {
        setPhase('error');
        setPhaseNote('init failed');
        addMessage('system', `Bridge init failed: ${(error as Error).message}`);
      }
    })();

    return () => {
      flushDraftNow();
      bridge.stop();
      bridgeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      setSpinnerTick(s => (s + 1) % 10);
      if (!busy || phase === 'streaming' || phase === 'error' || phase === 'ready') {
        return;
      }
    }, 100);

    return () => clearInterval(timer);
  }, [busy, phase]);

  useInput((key, inkKey) => {
    if (inkKey.ctrl && key.toLowerCase() === 'o') {
      setTraceOn(prev => {
        const next = !prev;
        pushTrace(`trace=${next ? 'on' : 'off'}`);
        return next;
      });
      return;
    }

    if (inkKey.ctrl && key.toLowerCase() === 'p') {
      setStreamPanelOn(prev => {
        const next = !prev;
        pushTrace(`stream_panel=${next ? 'on' : 'off'}`);
        return next;
      });
      return;
    }

    if (inkKey.ctrl && key.toLowerCase() === 'c') {
      exit();
    }
  });

  const runSlash = async (raw: string) => {
    const bridge = bridgeRef.current;
    if (!bridge) {
      addMessage('system', 'Bridge is not available.');
      return;
    }

    try {
      busyRef.current = true;
      setBusy(true);
      setPhase('thinking');
      setPhaseNote(raw);

      const result = await bridge.call<SlashResult>('slash', { raw, config_path: 'agent_config.json' });
      setState(result.state);
      for (const line of result.messages || []) {
        addMessage('system', line);
      }
      if (result.exit) {
        exit();
      }
      setIdle();
    } catch (error) {
      setPhase('error');
      setPhaseNote('slash failed');
      addMessage('system', `Slash command failed: ${(error as Error).message}`);
      busyRef.current = false;
      setBusy(false);
    }
  };

  const runChat = async (text: string) => {
    const bridge = bridgeRef.current;
    if (!bridge) {
      addMessage('system', 'Bridge is not available.');
      return;
    }

    try {
      busyRef.current = true;
      setBusy(true);
      setPhase('thinking');
      setPhaseNote('thinking');
      draftBufferRef.current = '';
      streamRawRef.current = '';
      liveAssistantIdRef.current = null;

      const result = await bridge.call<ChatResult>('chat', { message: text });
      setState(result.state);

      if (result.pipeline) {
        const liveId = liveAssistantIdRef.current;
        if (liveId) {
          setMessages(prev => prev.filter(message => message.id !== liveId));
          liveAssistantIdRef.current = null;
        }
        for (const turn of result.turns || []) {
          addMessage('assistant', `[${turn.agent}] ${turn.text}`);
        }
      } else if (typeof result.assistant === 'string') {
        finalizeLiveAssistantMessage(result.assistant);
      }

      setIdle();
    } catch (error) {
      flushDraftNow();
      stopLiveAssistantMessage();
      setPhase('error');
      setPhaseNote('chat failed');
      addMessage('system', `Chat failed: ${(error as Error).message}`);
      busyRef.current = false;
      setBusy(false);
    }
  };

  submitHandlerRef.current = async (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) {
      return;
    }

    addMessage('user', trimmed);

    if (trimmed.startsWith('/')) {
      const lower = trimmed.toLowerCase();
      if (lower === '/help' || lower === '/?') {
        addMessage('system', formatHelp());
        return;
      }

      if (lower === '/clear') {
        setMessages([]);
        draftBufferRef.current = '';
        streamRawRef.current = '';
        liveAssistantIdRef.current = null;
        addMessage('system', 'Transcript cleared. Session state is unchanged.');
        pushTrace('local_clear');
        return;
      }

      if (lower === '/doctor') {
        await runDoctor();
        return;
      }

      if (lower === '/bug' || lower.startsWith('/bug ')) {
        try {
          await runBugReport(trimmed);
        } catch (error) {
          addMessage('system', `Bug report failed: ${(error as Error).message}`);
        }
        return;
      }

      if (lower === '/trace' || lower.startsWith('/trace ')) {
        const parts = trimmed.split(/\s+/);
        const arg = (parts[1] || '').toLowerCase();
        let next = traceOnRef.current;
        if (!arg) {
          next = !traceOnRef.current;
        } else if (['on', '1', 'true', 'yes'].includes(arg)) {
          next = true;
        } else if (['off', '0', 'false', 'no'].includes(arg)) {
          next = false;
        } else {
          addMessage('system', 'Usage: /trace [on|off]');
          return;
        }
        setTraceOn(next);
        addMessage('system', `Trace ${next ? 'enabled' : 'disabled'}.`);
        pushTrace(`trace=${next ? 'on' : 'off'}`);
        return;
      }

      if (lower === '/status') {
        const bridge = bridgeRef.current;
        if (!bridge) {
          addMessage('system', 'Bridge is not available.');
          return;
        }
        try {
          const snap = await bridge.call<StateResult>('state', {});
          setState(snap);
          addMessage('system', formatStatus(snap));
        } catch (error) {
          addMessage('system', `Status failed: ${(error as Error).message}`);
        }
        return;
      }

      await runSlash(trimmed);
      return;
    }

    await runChat(trimmed);
  };

  const onSubmit = useCallback(async (value: string) => {
    await submitHandlerRef.current(value);
  }, []);

  const statusText = useMemo(() => {
    const SPINNERS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    const spin = (busy && phase !== 'ready' && phase !== 'error') ? `${SPINNERS[spinnerTick]} ` : '';
    const tools = state.tool_count != null ? `tools:${state.tool_count}` : '';
    const skills = state.skill_count != null ? `skills:${state.skill_count}` : '';
    return `${spin}${phase} · ${phaseNote} · agent:${state.active} · msgs:${state.msg_count} · ${tools} · ${skills} · trace:${traceOn ? 'on' : 'off'} · raw:${streamPanelOn ? 'on' : 'off'} · Ctrl+O/P`;
  }, [busy, phase, phaseNote, spinnerTick, state.active, state.msg_count, state.tool_count, state.skill_count, streamPanelOn, traceOn]);

  const completedMessages = messages.filter(m => !m.streaming);
  const liveMessages = messages.filter(m => m.streaming);

  return (
    <>
      <Static items={completedMessages}>
        {m => (
          <Box key={m.id} flexDirection="column" paddingX={1}>
            <MessageRow message={m} rawOn={streamPanelOn} />
          </Box>
        )}
      </Static>
      <Box flexDirection="column" paddingX={1}>
        {liveMessages.map(m => (
          <MessageRow key={m.id} message={m} rawOn={streamPanelOn} />
        ))}
        {liveMessages.length > 0 && <Newline />}
        <PromptInput onSubmit={onSubmit} />
        <Box borderStyle="single" borderColor="gray">
          <Text color={phaseColor[phase]}>{statusText}</Text>
        </Box>
      </Box>
    </>
  );
};

render(<App />);
