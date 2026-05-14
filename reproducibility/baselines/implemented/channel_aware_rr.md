# Channel-aware round-robin

## Status

Self-contained implementation in the shared B2CRoI-H(Q) simulator; no third-party official code is vendored.

## Selection rule summary

Keeps cyclic service but prioritizes loops whose channels are currently in the good state when possible.

## Role in evaluation

Channel-aware baseline under burst-loss networks.

## Reproducibility note

See `../BASELINE_REGISTRY.csv` for implementation location and third-party-code status. If this baseline is later replaced by an official repository implementation, record the repository, commit, license, command, and metric mapping in `../../third_party_sources/sources.lock`.
