# TUI palettes

Each bundled theme defines its own speaker accents, reasoning color, syntax
colors, input border, and separate code and tool surfaces. Light themes use dark
ink on pale surfaces; dark themes retain their namesake palette with readable
secondary text and visible borders.

`canvas` records the intended terminal background. The TUI leaves the terminal's
main background unchanged. Code blocks use `canvasInset`; tool cards use
`canvasSubtle`. For the intended appearance, match your terminal background to
that theme's `canvas` value. User themes in `~/.logician/themes` take precedence
over bundled files with the same name.

The readability tests check normal and secondary block text, syntax colors, and
status colors at a minimum 4.5:1 contrast against both block surfaces. Borders
are checked at 2.5:1. Both truecolor and the renderer's 256-color conversion are
tested. Terminal dimming, custom terminal palettes, and externally supplied ANSI
output can affect the final appearance beyond these checks.

Run from the repository root:

```sh
bun test apps/tui/src/__tests__/theme-readability.test.ts
```

When changing a palette, keep success/error/diff meanings consistent and adjust
foregrounds together with their surfaces. Syntax rendering uses the active
palette; the runtime's cached highlighting remains independent of theme changes.
