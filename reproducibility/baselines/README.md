# Baseline registry for B2CRoI-H(Q)

This directory documents the comparison methods used by the B2CRoI-H(Q) reproducibility package.

## Scope

The current package uses **self-contained baseline implementations** inside the shared simulator, mainly in:

- `../../scripts/b2croi_v8q_benchmark.py`
- `../../scripts/b2croi_v8q_stress_n.py`
- `../../scripts/b2croi_v8_ablation.py`

No official third-party GitHub implementation is currently vendored or executed by the reproducibility pipeline. Therefore, the study should describe these methods as **representative** or **literature-inspired** baselines unless a specific official implementation, commit, and license are added later.

## Why self-contained baselines?

All policies are evaluated under the same data windows, network traces, bandwidth constraints, prediction module, and metric definitions. This avoids unfair differences caused by separate simulators, incompatible data preprocessing, or hidden hyperparameters.

## Do not overclaim

Recommended wording:

> representative AoI-, VoI-, channel-aware, and event-triggered scheduling baselines

Avoid unless official implementations are added and verified:

> state-of-the-art implementations

## Files

- `BASELINE_REGISTRY.csv`: machine-readable baseline list.
- `implemented/*.md`: short human-readable descriptions.
- `../third_party_sources/sources.lock`: placeholder for future official repositories, commits, and licenses.
## Fidelity audit

See `FIDELITY_AUDIT.md` for a conservative method-by-method disclosure. No baseline in this package is claimed to be a 100% official reproduction of a named third-party reference implementation.

