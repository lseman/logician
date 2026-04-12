---
name: Edit Block
description: Use for precise block-level edits and targeted code replacements inside files.
preferred_tools:
  - apply_edit_block
callable: true
scripts:
  - scripts/edit_block.py
---

See `scripts/edit_block.py` for implementation and usage details. This skill is marked as
callable: the loader will not import the implementation at discovery time. To run the
implementation, use the loader's `call_skill()` API with `allow_exec=True`.
