# docs/research - the rdloop research team's shared record

| File / dir | Written by | What it is |
|---|---|---|
| `lever_ledger.md` | seeded from `docs/experiment_log.csv`; rdloop appends per cycle | every lever tried / built, with its outcome; read first by every cycle |
| `literature.md`, `literature.jsonl` | the rdloop supervisor (from the scout's verified sources) | sources collected across cycles |
| `<C>/` | the rdloop supervisor at the end of cycle `<C>` | copies of `01_audit.md`, `02_sota.md`, `plan_agreed.md`, `FINAL_SUMMARY.md`, `debate/*.md` |

The working files of a cycle (state, prompts, tests, smoke runs) live in `rdloop/<C>/`.
How to run a cycle: `docs/research_agent.md` (use cases + workbook). Operator details:
`scripts/rdloop/README.md`.
