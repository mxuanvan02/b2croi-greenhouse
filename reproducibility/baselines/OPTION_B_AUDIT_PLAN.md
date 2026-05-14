# Option B baseline fidelity upgrade plan

Goal: upgrade baseline fidelity from conservative representative implementations toward paper-faithful reproductions where evidence supports it. No baseline is upgraded by wording alone.

## Gates

A baseline can be marked `paper-faithful` only after all applicable gates pass:

1. target paper/reference selected;
2. equation or algorithm block extracted;
3. implementation mapping documented;
4. hyperparameters and adaptation choices documented;
5. official repository/commit/license recorded if available;
6. toy-case decision test passes;
7. manuscript wording updated to match the verified fidelity level.

## Immediate targets

1. `round_robin`, `max_aoi`, and `oracle`: add deterministic toy decision tests.
2. `generic_voi`: audit against Molin et al. (Automatica 2019), Ayan et al. (ICCPS 2019), and Wang et al. (CDC 2021) references already present in `manuscript/compag/refs.bib`.
3. `error_trigger`: audit against event-triggered control references already cited in the manuscript.
4. `channel_aware_rr`: keep as channel-aware cyclic baseline unless a specific official method is selected.
5. `cvoi_sf`: either identify an exact target paper/equation or keep the current adapted-comparator disclosure.

## Current conclusion

At this checkpoint, only classical/definition baselines can be considered exact within the shared simulator. VoI-, channel-aware-, event-triggered-, and CVoI-SF-style baselines remain under audit and must not be called official SOTA reproductions.
