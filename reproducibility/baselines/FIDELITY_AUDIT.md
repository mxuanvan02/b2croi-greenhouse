# Baseline fidelity audit for B2CRoI-H(Q)

This audit is intentionally conservative and transparent.

Current package status: **no official third-party GitHub baseline implementation is vendored or executed**. The comparison methods are classical or representative/literature-inspired baselines implemented in a shared simulator.

## Summary table

| Baseline | Fidelity status | Official third-party code used? | Safe manuscript wording | Unsafe wording |
|---|---|---:|---|---|
| `round_robin` | exact classical cyclic implementation | No | standard round-robin baseline | official SOTA reproduction |
| `max_aoi` | exact classical largest-AoI implementation in this simulator | No | largest-AoI baseline | exact reproduction of a specific AoI paper unless cited/audited |
| `channel_aware_rr` | shared-simulator, channel-aware cyclic adaptation | No | channel-aware cyclic baseline | official channel-aware method reproduction |
| `error_trigger` | shared-simulator, error-triggered representative baseline | No | error-triggered representative baseline | exact reproduction of a named event-triggered-control method |
| `generic_voi` | literature-inspired VoI scoring baseline | No | generic VoI-inspired baseline | exact reproduction of a named VoI paper |
| `cvoi_sf` | adapted constrained VoI/service-floor baseline | No | constrained VoI/service-floor baseline | official CVoI-SF implementation |
| `ar1_growth_voi` | stress-test predictive VoI-style baseline | No | predictive VoI stress-test baseline | official SOTA method |
| `oracle` | non-causal upper-bound/reference | No | oracle upper-bound/reference | deployable competing method |

## Reviewer-facing disclosure text

Recommended wording:

> All baselines were implemented in a shared simulator to ensure identical data windows, channel traces, bandwidth budgets, prediction modules, and metric definitions. Classical baselines such as round-robin and largest-AoI are implemented directly. The VoI-, channel-aware, and event-triggered comparators are representative/literature-inspired implementations rather than official third-party code reproductions. No official external baseline repository is vendored or executed in this package.

## Method-level notes

### `round_robin`
- Implementation: `../../scripts/b2croi_v8q_benchmark.py`.
- Rule: select loops cyclically according to the bandwidth budget.
- Fidelity: exact for the standard cyclic scheduling rule.

### `max_aoi`
- Implementation: `../../scripts/b2croi_v8q_benchmark.py`.
- Rule: prioritize loops with the largest current age of information.
- Fidelity: exact for largest-age-first scheduling as defined in this simulator; not claimed to match every AoI paper variant.

### `channel_aware_rr`
- Implementation: `../../scripts/b2croi_v8q_benchmark.py`.
- Rule: retain cyclic service while preferring loops whose channels are in favorable states.
- Fidelity: representative shared-simulator baseline, not official external-code reproduction.

### `error_trigger`
- Implementation: `../../scripts/b2croi_v8q_benchmark.py`.
- Rule: prioritize loops with larger estimation mismatch/error.
- Fidelity: representative event/error-triggered baseline, not a named-paper reproduction.

### `generic_voi`
- Implementation: `../../scripts/b2croi_v8q_benchmark.py`.
- Rule: rank loops by a generic value-of-information score combining estimation mismatch and age.
- Fidelity: literature-inspired, not a named official VoI implementation.

### `cvoi_sf`
- Implementation: `../../scripts/b2croi_v8q_benchmark.py`.
- Rule: extend generic VoI-style scoring with safety/service-floor terms.
- Fidelity: adapted constrained VoI/service-floor comparator designed for common-simulator fairness; not official CVoI-SF code from a prior paper.

### `ar1_growth_voi`
- Implementation: `../../scripts/b2croi_v8q_stress_n.py`.
- Rule: predictive growth/VoI-style comparator used in N/heterogeneity stress testing.
- Fidelity: stress-test representative comparator, not official SOTA reproduction.

### `oracle`
- Implementation: `../../scripts/b2croi_v8q_benchmark.py`.
- Rule: use realized/privileged information to define an upper-bound-like reference.
- Fidelity: non-causal reference, not a deployable competitor.

## Requirements before claiming official SOTA reproduction

Before any baseline is described as an official/faithful reproduction, add: original paper citation, repository URL, commit or release, license check, environment file, unmodified run log, adaptation patch if any, metric mapping, and at least one toy-case decision-equivalence test.

## Option B audit tracking

See `FIDELITY_TARGETS.csv` and `OPTION_B_AUDIT_PLAN.md` for the active paper-faithful reproduction upgrade plan.

## Audit update: source-code and literature findings

- Exact-title repository searches did not identify official GitHub implementations for the audited VoI/event-triggered/channel-aware target papers. The status is therefore `not found/not recorded`, not proof that no repository exists.
- DOI metadata was corrected in `manuscript/compag/refs.bib` for Ayan et al. 2019 and Wang et al. 2021 according to the audit evidence.
- `reproducibility/tests/test_baseline_decisions.py` verifies deterministic behavior for exact classical baselines and documents adapted comparator behavior.
- `channel_aware_rr` is retained as a representative burst-aware cyclic comparator in the current shared simulator. Its current channel-success term is shared across loops, so it should not be described as a fully per-loop channel-state scheduler.
