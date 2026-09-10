<!--
Sync Impact Report
==================
Version change: (unfilled template scaffold) -> 1.0.0
Bump rationale: Initial ratification. MAJOR version 1.0.0 establishes the first binding
governance document for this project; no prior principles existed, so there is no
backward-compatibility surface to preserve.

Modified principles:
- [PRINCIPLE_1_NAME] -> I. Task List Is the Contract (Superpowers Workflow)
- [PRINCIPLE_2_NAME] -> II. Spec-Scoped Test-First (NON-NEGOTIABLE)
- [PRINCIPLE_3_NAME] -> III. Simplicity and Reuse (Ponytail Full Mode)
- [PRINCIPLE_4_NAME] -> removed (user supplied three principles; unused template slot retired)
- [PRINCIPLE_5_NAME] -> removed (user supplied three principles; unused template slot retired)

Added sections:
- Development Workflow (fills [SECTION_2_NAME])
- Quality Gates (fills [SECTION_3_NAME])
- Governance (filled from template slot)

Removed sections: none (all template slots filled or retired as noted above)

Follow-up TODOs: none - all placeholders resolved.
-->

# DIPR Project Constitution

## Core Principles

### I. Task List Is the Contract (Superpowers Workflow)

Implementation of any task list MUST follow the Superpowers workflow, in this order:
worktree -> TDD (red-green-refactor) -> subagent-driven execution -> code review ->
finish-branch. Once a task list is approved, it is the binding contract for execution:
agents MUST NOT re-plan, re-scope, or rewrite tasks during implementation. If a task
proves infeasible or incorrect, work MUST stop and the deviation MUST be surfaced to
the user for an explicit task-list amendment before proceeding.

**Rationale**: A single approved artifact (the task list) eliminates scope drift and
parallel re-planning; a fixed pipeline makes every implementation run auditable and
reproducible.

### II. Spec-Scoped Test-First (NON-NEGOTIABLE)

All implementation MUST proceed test-first using strict red-green-refactor: write a
failing test, make it pass with the minimal implementation, then refactor under green.
Tests MUST strictly cover only the acceptance criteria defined in the current spec
task. Tests for speculative features, hypothetical requirements, or acceptance criteria
belonging to other tasks MUST NOT be written.

**Rationale**: Tests anchored to acceptance criteria keep the suite meaningful and the
red phase honest; speculative tests encode requirements nobody approved and create
false obligations for future work.

### III. Simplicity and Reuse (Ponytail Full Mode)

Ponytail (full mode) MUST be applied at the plan stage and again before every commit:
prefer YAGNI (do not build what the current spec does not require), reuse existing
code in the repository over writing new code, and avoid new dependencies unless the
task cannot be completed without them. Ponytail MUST NEVER be used to skip or weaken
input validation, security checks, accessibility requirements, or data-loss handling -
those remain non-negotiable regardless of simplicity pressure.

**Rationale**: Simplicity and reuse reduce maintenance surface and review cost, while
the never-skip list guarantees minimalism does not become a license to ship unsafe,
inaccessible, or destructive code.

## Development Workflow

Every implementation run MUST execute the following pipeline stages in order:

1. **Worktree**: Create an isolated worktree/branch before any code change. Work MUST
   NOT be performed directly on the shared base branch.
2. **TDD (red-green-refactor)**: For each task, follow Principle II - failing test
   first, minimal implementation, refactor under green.
3. **Subagent-driven execution**: Execute tasks from the approved task list via
   subagents. Each subagent MUST receive its task verbatim and MUST NOT re-plan it
   (Principle I).
4. **Code review**: Run a code review after implementation and address CRITICAL and
   HIGH findings before the run may finish.
5. **Finish-branch**: Complete the branch (merge or hand-off) only after all review
   gates pass.

## Quality Gates

- **Plan-stage gate**: A ponytail (full mode) review MUST pass before any plan is
  approved for task generation.
- **Pre-commit gate**: Before every commit, apply the ponytail checks - no speculative
  code (YAGNI), existing code reused where possible, no unnecessary new dependencies -
  and confirm that validation, security, accessibility, and data-loss handling remain
  intact.
- **Test-scope gate**: Each task's tests MUST be traceable to that task's acceptance
  criteria; untraceable tests MUST be removed.
- **Review gate**: Code review MUST complete with no unresolved CRITICAL findings
  before finish-branch.

## Governance

This constitution supersedes all other development practices for this project. Where a
practice document, template, or agent instruction conflicts with this constitution, the
constitution wins.

- **Amendment procedure**: Amendments MUST be proposed explicitly, documented with a
  version bump and a Sync Impact Report, and applied via the `/speckit-constitution`
  command. Silent amendments MUST NOT be made.
- **Versioning policy**: This document follows semantic versioning - MAJOR for backward
  incompatible governance or principle removals/redefinitions, MINOR for new principles
  or materially expanded guidance, PATCH for clarifications and non-semantic fixes.
- **Compliance review**: All plans, task executions, and code reviews MUST verify
  compliance with these principles. Any deviation MUST be surfaced to the user and
  resolved by amendment or correction - deviations MUST NOT be absorbed silently.
  Runtime development guidance is provided by the Spec Kit command set
  (`/speckit-plan`, `/speckit-tasks`, `/speckit-implement`) under this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-09-09 | **Last Amended**: 2026-09-09
