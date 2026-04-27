# Review scopes in v1.1.2x

`review-build` now supports two review scopes:

```bash
acs review-build ... --review-scope candidates
acs review-build ... --review-scope all-events --top-k 0
```

`candidates` is the historical default and writes to `run/review`.
`all-events` writes to `run/review_all_events` by default and renders records from
`events.jsonl` for manual SETI veto/inspection. Set `--top-k 0` or omit `--top-k`
for an unbounded all-event review; provide a positive `--top-k` to inspect only the
highest-scoring events.

The all-event mode does not change search results. It only changes which records
are rendered during review.
