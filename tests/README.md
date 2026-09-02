# Tests

Unit and integration tests for strainphase.

## Running

```bash
pytest tests/                                   # all tests
pytest tests/ -v                                # verbose
pytest tests/test_core.py::TestHaplotyperConfig # one class
```

## Files

- `test_core.py` — data structures, graph init, EM, post-processing, within-sample linking
- `test_parsing.py` — VCF loading and the CIGAR walk that reads indels off reads
- `test_sv_encoding.py` — structural-variant pseudo-SNV encoding and reconciliation
- `test_window_linking_rework.py` — the identity gate stack, marker sets, and read-overlap linking
- `test_track_merge.py` — merging within-sample tracks across samples into lineages
- `test_longitudinal.py` — cross-timepoint anchor panels, rescue, and published-table shape
- `test_linking_scenarios.py` — linking cases reduced from real runs, scored on fragmentation vs error
- `test_real_data_regressions.py` — xfail specifications built from real cohort output
- `test_memory_offload.py` — read spilling and window batching must not change any output value
- `util_io.py` — helpers that build tiny VCFs, FASTAs, and BAMs for the tests
