// ── GSD Bridge — Phase lifecycle management ─────────────────────────────────
// Manages phase directories under .planning/phases/ and .planning/milestones/
// Handles phase creation, planning, execution tracking, and completion.

import fs from 'node:fs';
import path from 'node:path';

const PHASES_DIR = '.planning/phases';
const MILESTONES_DIR = '.planning/milestones';

interface PhaseInfo {
  phaseId: string;
  paddedPhase: string;
  phaseName: string;
  phaseSlug: string;
  phaseDir: string;
  roadmapLine: number;
  status: 'unplanned' | 'planned' | 'in-progress' | 'complete' | 'halted';
}

export function getPhaseDir(cwd: string, phaseId: string): string {
  const padded = padPhase(phaseId);
  const dir = path.join(cwd, PHASES_DIR, padded);
  if (!fs.existsSync(dir)) {
    throw new Error(`Phase ${phaseId} not found. Run /gsd:discuss-phase ${phaseId} first.`);
  }
  return dir;
}

function padPhase(phaseId: string): string {
  const num = parseInt(phaseId, 10);
  return String(num).padStart(2, '0');
}

export function createPhase(cwd: string, phaseId: string, name: string): string {
  const padded = padPhase(phaseId);
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const phaseDir = path.join(cwd, PHASES_DIR, padded);
  const nameDir = path.join(phaseDir, slug);

  if (!fs.existsSync(phaseDir)) {
    fs.mkdirSync(phaseDir, { recursive: true });
  }
  if (!fs.existsSync(nameDir)) {
    fs.mkdirSync(nameDir, { recursive: true });
  }

  // Create initial CONTEXT.md
  const contextMd = `# Phase ${padded}: ${name}\n\n## Context\n\nNo context captured yet. Run discuss-phase first.\n\n`;
  fs.writeFileSync(path.join(nameDir, 'CONTEXT.md'), contextMd);

  return nameDir;
}

export function listPhases(cwd: string): PhaseInfo[] {
  const phasesDir = path.join(cwd, PHASES_DIR);
  if (!fs.existsSync(phasesDir)) {
    return [];
  }

  const result: PhaseInfo[] = [];
  const phaseDirs = fs.readdirSync(phasesDir).filter(d => /^\d{2}$/.test(d));

  for (const padded of phaseDirs) {
    const phaseDir = path.join(phasesDir, padded);
    const phaseNum = parseInt(padded, 10);
    const subdirs = fs.readdirSync(phaseDir).filter(d => !d.endsWith('.md'));

    for (const slug of subdirs) {
      const phasePath = path.join(phaseDir, slug);
      const summaryPath = path.join(phasePath, 'SUMMARY.md');
      const status: PhaseInfo['status'] = fs.existsSync(summaryPath)
        ? fs.readFileSync(summaryPath, 'utf-8').includes('status: complete')
          ? 'complete'
          : 'in-progress'
        : 'unplanned';

      result.push({
        phaseId: String(phaseNum),
        paddedPhase: padded,
        phaseName: slug,
        phaseSlug: slug,
        phaseDir: phasePath,
        roadmapLine: phaseNum,
        status,
      });
    }
  }

  return result.sort((a, b) => parseInt(a.paddedPhase, 10) - parseInt(b.paddedPhase, 10));
}

export function getPhaseStatus(cwd: string, phaseId: string): string {
  const phases = listPhases(cwd);
  const phase = phases.find(p => p.phaseId === phaseId || p.paddedPhase === padPhase(phaseId));
  if (!phase) return 'not-found';
  return phase.status;
}

export function planPhase(cwd: string, phaseId: string, planContent: string): void {
  const phaseDir = getPhaseDir(cwd, phaseId);
  fs.writeFileSync(path.join(phaseDir, 'PLAN.md'), planContent);
}

export function completePlanPhase(cwd: string, phaseId: string, summary: string): void {
  const phaseDir = getPhaseDir(cwd, phaseId);
  fs.writeFileSync(path.join(phaseDir, 'SUMMARY.md'), summary);
}
