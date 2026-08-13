// ── GSD Bridge — Logician extension for GSD Core workflows ──────────────────
// Registers all GSD slash commands and provides workflow execution adapters.
// Maps GSD's .planning/ filesystem operations to Logician's extension API.

import type { ExtensionAPI, ExtensionContext, RegisteredCommand } from '../../packages/agent-core/src/extensions/types.ts';
import * as state from './src/state.ts';
import * as phase from './src/phase.ts';

const GSD_COMMANDS: RegisteredCommand[] = [
  // ── Project Lifecycle ─────────────────────────────────────────────
  {
    name: '/gsd:new-project',
    description: 'Initialize a new project with deep context gathering and PROJECT.md',
    usage: '[--auto]',
    acceptsArgs: true,
    handler: async (args: string, ctx: { ui: ExtensionContext['ui'], sessionId: string, cwd: string }) => {
      const auto = args.includes('--auto');
      // Read and adapt new-project workflow
      const workflowPath = `repos/gsd-core/gsd-core/workflows/new-project.md`;
      return `GSDBRIDGE:workflow:new-project${auto ? ':auto' : ''}:${workflowPath}`;
    },
  },
  {
    name: '/gsd:onboard',
    description: 'Onboard an existing codebase through GSD planning setup',
    usage: '[--auto]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:workflow:onboard:${args.includes('--auto') ? 'auto' : 'interactive'}`;
    },
  },
  {
    name: '/gsd:next',
    description: 'Smart entry — detect project state and route to the right next GSD action',
    usage: '',
    acceptsArgs: false,
    handler: async (_args, ctx) => {
      const hasProject = ctx.cwd && require('node:fs').existsSync(require('node:path').join(ctx.cwd, '.planning/PROJECT.md'));
      const hasState = ctx.cwd && require('node:fs').existsSync(require('node:path').join(ctx.cwd, '.planning/STATE.md'));
      const phasesDir = ctx.cwd && require('node:fs').existsSync(require('node:path').join(ctx.cwd, '.planning/phases'));
      if (!hasProject) return 'No project initialized. Run /gsd:new-project first.';
      if (!phasesDir) return 'No phases found. Run /gsd:progress to start planning.';
      const phases = phase.listPhases(ctx.cwd);
      const nextPhase = phases.find(p => p.status === 'unplanned');
      if (nextPhase) return `Next unplanned phase: ${nextPhase.phaseId} (${nextPhase.phaseName}). Run /gsd:discuss-phase ${nextPhase.phaseId}`;
      const inProgress = phases.find(p => p.status === 'in-progress');
      if (inProgress) return `In-progress: ${inProgress.phaseId}. Run /gsd:progress or /gsd:execute-phase`;
      return 'All phases complete. Run /gsd:progress for summary.';
    },
  },
  // ── Phase Workflows ───────────────────────────────────────────────
  {
    name: '/gsd:discuss-phase',
    description: 'Gather phase context through adaptive questioning before planning',
    usage: '<phase> [--all] [--auto] [--chain] [--batch] [--analyze] [--text] [--power] [--assumptions]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      const phaseNum = args.split(' ')[0];
      return `GSDBRIDGE:workflow:discuss-phase:${phaseNum}:${args.includes('--power') ? 'power' : 'standard'}`;
    },
  },
  {
    name: '/gsd:plan-phase',
    description: 'Create detailed phase plan (PLAN.md) with verification loop',
    usage: '[phase] [--auto] [--research] [--skip-research] [--gaps] [--skip-verify] [--mvp] [--tdd]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      const phaseNum = args.split(' ')[0];
      return `GSDBRIDGE:workflow:plan-phase:${phaseNum}:${args}`;
    },
  },
  {
    name: '/gsd:execute-phase',
    description: 'Execute all plans in a phase with wave-based parallelization',
    usage: '<phase> [--auto] [--wave <n>]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      const phaseNum = args.split(' ')[0];
      return `GSDBRIDGE:workflow:execute-phase:${phaseNum}`;
    },
  },
  {
    name: '/gsd:verify-work',
    description: 'Validate built features against requirements',
    usage: '<phase>',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      const phaseNum = args.split(' ')[0];
      return `GSDBRIDGE:workflow:verify-work:${phaseNum}`;
    },
  },
  {
    name: '/gsd:ship',
    description: 'Create PR, run review, and prepare for merge after verification passes',
    usage: '[phase number or milestone]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:workflow:ship:${args}`;
    },
  },
  // ── Quick Tasks ───────────────────────────────────────────────────
  {
    name: '/gsd:quick',
    description: 'Execute a quick task with GSD guarantees (atomic commits, state tracking) but skip optional agents',
    usage: '[list | status <slug> | resume <slug> | --full] [--validate] [--discuss] [--research] [task description]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      if (args.startsWith('list')) return `GSDBRIDGE:quick:list`;
      if (args.startsWith('status ')) return `GSDBRIDGE:quick:status:${args.slice(7)}`;
      if (args.startsWith('resume ')) return `GSDBRIDGE:quick:resume:${args.slice(7)}`;
      return `GSDBRIDGE:quick:run:${args}`;
    },
  },
  // ── Project Management ────────────────────────────────────────────
  {
    name: '/gsd:progress',
    description: 'Check progress, advance workflow, or dispatch freeform intent',
    usage: '[<phase>] [--verbose] [--json]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:progress:${args}`;
    },
  },
  {
    name: '/gsd:stats',
    description: 'Display project statistics — phases, plans, requirements, git metrics',
    usage: '',
    acceptsArgs: false,
    handler: async (_args, ctx) => {
      const phases = phase.listPhases(ctx.cwd);
      const completed = phases.filter(p => p.status === 'complete').length;
      const inProgress = phases.filter(p => p.status === 'in-progress').length;
      const unplanned = phases.filter(p => p.status === 'unplanned').length;
      return `Project Stats: ${phases.length} phases | ${completed} done | ${inProgress} in progress | ${unplanned} unplanned`;
    },
  },
  {
    name: '/gsd:phase',
    description: 'CRUD for phases in ROADMAP.md — add, insert, remove, or edit phases',
    usage: '<add|remove|list|complete> [args...]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:phase:${args}`;
    },
  },
  {
    name: '/gsd:milestone-summary',
    description: 'Generate a comprehensive project summary from milestone artifacts',
    usage: '',
    acceptsArgs: false,
    handler: async (_args, ctx) => {
      return `GSDBRIDGE:milestone:summary`;
    },
  },
  // ── Configuration ─────────────────────────────────────────────────
  {
    name: '/gsd:config',
    description: 'Configure GSD settings — workflow toggles, advanced knobs',
    usage: '<get|set> [key] [value]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:config:${args}`;
    },
  },
  {
    name: '/gsd:settings',
    description: 'Configure GSD workflow toggles and model profile',
    usage: '[list | get <key> | set <key> <value>]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:settings:${args}`;
    },
  },
  // ── Review & Audit ────────────────────────────────────────────────
  {
    name: '/gsd:code-review',
    description: 'Review source files changed during a phase for bugs and quality',
    usage: '[phase]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:code-review:${args}`;
    },
  },
  {
    name: '/gsd:ui-review',
    description: 'Retroactive 6-pillar visual audit of implemented frontend code',
    usage: '[phase]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:ui-review:${args}`;
    },
  },
  {
    name: '/gsd:audit-fix',
    description: 'Autonomous audit-to-fix pipeline — find issues, classify, fix, test',
    usage: '[target]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:audit-fix:${args}`;
    },
  },
  // ── Memory & Context ──────────────────────────────────────────────
  {
    name: '/gsd:mempalace-capture',
    description: 'File a phase artifact into MemPalace; mirror decision facts into temporal KG',
    usage: '<artifact-path>',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:mempalace:capture:${args}`;
    },
  },
  {
    name: '/gsd:mempalace-recall',
    description: 'Recall decisions, patterns, and surprises from MemPalace',
    usage: '[query]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:mempalace:recall:${args || 'all'}`;
    },
  },
  {
    name: '/gsd:capture',
    description: 'Capture ideas, tasks, notes, and seeds to their destination',
    usage: '<text>',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:capture:${args}`;
    },
  },
  // ── Workstreams ───────────────────────────────────────────────────
  {
    name: '/gsd:workstreams',
    description: 'Manage parallel workstreams — list, create, switch, status, complete',
    usage: '<list | create | switch | status | complete> [args...]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:workstreams:${args}`;
    },
  },
  {
    name: '/gsd:workspace',
    description: 'Manage GSD workspaces — create, list, or remove isolated environments',
    usage: '<create | list | remove> [name]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:workspace:${args}`;
    },
  },
  // ── Thread & Pause ────────────────────────────────────────────────
  {
    name: '/gsd:thread',
    description: 'Manage persistent context threads for cross-session work',
    usage: '<list | create | switch | close> [name]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:thread:${args}`;
    },
  },
  {
    name: '/gsd:pause-work',
    description: 'Create context handoff when pausing work mid-phase',
    usage: '[reason]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:pause:${args || 'unspecified'}`;
    },
  },
  {
    name: '/gsd:resume-work',
    description: 'Resume work from previous session with full context restoration',
    usage: '[thread | phase]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:resume:${args || 'latest'}`;
    },
  },
  // ── Additional Commands ───────────────────────────────────────────
  {
    name: '/gsd:explore',
    description: 'Socratic ideation and idea routing — think through ideas before committing',
    usage: '[topic]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:explore:${args}`;
    },
  },
  {
    name: '/gsd:sketch',
    description: 'Sketch UI/design ideas with throwaway HTML mockups',
    usage: '[description]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:sketch:${args}`;
    },
  },
  {
    name: '/gsd:spike',
    description: 'Spike an idea through experiential exploration',
    usage: '[topic]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      return `GSDBRIDGE:spike:${args}`;
    },
  },
  {
    name: '/gsd:help',
    description: 'Show available GSD commands and usage guide',
    usage: '[command]',
    acceptsArgs: true,
    handler: async (args, ctx) => {
      if (args) return `GSD command: /gsd:${args}\nSee skills/gsd/README.md for full documentation.`;
      const cmds = [
        'Project: /gsd:new-project, /gsd:onboard, /gsd:next, /gsd:progress',
        'Phases: /gsd:discuss-phase, /gsd:plan-phase, /gsd:execute-phase, /gsd:verify-work, /gsd:ship',
        'Quick: /gsd:quick, /gsd:phase',
        'Mgmt: /gsd:stats, /gsd:config, /gsd:settings, /gsd:workstreams',
        'Review: /gsd:code-review, /gsd:ui-review, /gsd:audit-fix',
        'Memory: /gsd:capture, /gsd:mempalace-capture, /gsd:mempalace-recall',
        'Session: /gsd:thread, /gsd:pause-work, /gsd:resume-work',
        'Ideate: /gsd:explore, /gsd:sketch, /gsd:spike',
      ].join('\n');
      return `GSD Core Commands:\n${cmds}`;
    },
  },
];

export function registerGSDCommands(api: ExtensionAPI): void {
  for (const cmd of GSD_COMMANDS) {
    api.registerCommand(cmd);
  }
}
