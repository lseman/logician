# TUI dependency contracts

The TUI has two important seams:

1. `src/index.ts` is the composition root. It creates runtime and terminal adapters.
2. `src/app/` owns application orchestration. Presentation and foundation modules
   (`rendering`, `overlays`, `input`, `footer`, `status`, `state`, and `terminal`)
   must not import it.

The intended dependency direction is:

```text
index.ts → app → presentation modules
                 rendering → terminal primitives
```

Presentation modules may consume narrow data/formatting interfaces such as
`@logician/log-runtime/sessions` and `@logician/log-runtime/formatting`. They may
not reach into runtime application, configuration, context, reasoning, tooling,
trust, or developer-tool surfaces. Those integrations belong in `src/app/` or
the composition root.

All `@logician/*` imports must also be declared in `apps/tui/package.json` and
must use a subpath exported by the target package. Cross-workspace relative
imports are not package interfaces and are prohibited implicitly by this rule.

These contracts are executable in
`src/__tests__/architecture-contracts.test.ts` and run as part of the normal TUI
test suite.
