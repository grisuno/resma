# Second Brain

*Last synthesized: 2026-10-07 | 42 files | 4 concept pages | offline, zero tokens*

> Raw sources -> readmenator wiki -> links (Karpathy LLM Wiki Pattern, deterministic).
> Start here, then open one community page. Prefer grep over full reads.

## Vault Overview

The codebase centres on `resma_core.py`, `resma_observer.py`, `main5.py`. Architecturally it is 2 layers, dominant utility (38 files) across 4 import-based communities. Recorded risk surface: 0 security findings and 0 dependency cycles.

Surprising tissue lives between resma2: resma_core, resma2: monitor, root: 1 extracted cross-community imports and 10 inferred bridges. Follow `connections.json` sorted by strength before refactoring.

Open work clusters around documentation (62% file coverage), 0 security findings, 0 taint paths, and 5 suggested exploration questions in `queries.md`.

## Stats

| Metric | Value |
|--------|-------|
| Files | 42 |
| Symbols | 905 |
| Resolved imports | 18 |
| Languages | py, sh |
| Communities | 4 |
| Doc coverage | 62% (26/42 files) |
| Security findings | 0 |
| Estimated read cost | ~12714 tokens (chars/4, offline so $0) |

## Reading Order

1. Skim Stats and God Nodes below for blast radius.
2. Open the largest community page first, then follow Connections.
3. Use `queries.md` for the next question; log the answer there.

```
grep -rn '<keyword>' index.md community_*.md
readmenator query "<question>" --target readmenator_resma_r0fpdouq
```

## Concept Wiki

- [resma2: resma_core (8 files, cohesion 0.70)](./community_0_resma2_resma_core.md)
- [resma2: monitor (5 files, cohesion 0.62)](./community_1_resma2_monitor.md)
- [root (4 files, cohesion 1.00)](./community_2_root.md)
- [orphans (25 files, cohesion 0.00)](./community_3_orphans.md)

## God Nodes

| File | Score |
|------|-------|
| `resma2/resma_core.py` | 21.0 |
| `resma2/resma_observer.py` | 8.9 |
| `main5.py` | 8.1 |
| `garnier_nn.py` | 7.1 |
| `main.py` | 6.3 |

## Strongest Connections

- 1 -> 0: depends_on (strength 0.9, EXTRACTED)
- 1 -> 0: bridges (strength 0.7, INFERRED)
- 1 -> 0: bridges (strength 0.7, INFERRED)
- 1 -> 0: bridges (strength 0.7, INFERRED)
- 1 -> 0: bridges (strength 0.7, INFERRED)
- 1 -> 0: bridges (strength 0.7, INFERRED)
- 0 -> 2: shares_context (strength 0.5, INFERRED)
- 0 -> 3: shares_context (strength 0.5, INFERRED)
- 1 -> 2: shares_context (strength 0.5, INFERRED)
- 1 -> 3: shares_context (strength 0.5, INFERRED)

## Navigation Tips

- Obsidian Graph View works: every community page links back here.
- `connections.json` is machine-readable for GraphRAG pipelines.
- `REPORT.md` states what was extracted vs inferred and current limits.
- Regenerate offline: `readmenator . --rebuild` (no network, no tokens).
