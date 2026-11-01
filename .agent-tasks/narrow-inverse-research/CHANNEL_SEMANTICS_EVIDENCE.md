# BATADAL Channel-Semantics Evidence

**Evidence date:** 2026-10-11  
**Purpose:** Bound the claims encoded in the first channel/constraint manifest.

## Retrieved Source Claims

### Public C-Town Model

- **Source:** `rtaormina/epanetCPA`, `ctown_map.inp`, retrieved 2026-10-11.
  <https://github.com/rtaormina/epanetCPA/blob/master/ctown_map.inp>
- **Validated quotes:** `Units               LPS`; the model also contains
  explicit `T1`, `PU1`, and `V2` entries in its tank, pump, and valve sections.
- **Supported claim:** This public C-Town model uses the EPANET `LPS` unit system
  and contains public elements whose names resemble BATADAL channel suffixes.

### EPANET Unit Convention

- **Source:** US EPA Water Network Tool for Resilience documentation,
  `documentation/units.rst`, retrieved 2026-10-11.
  <https://github.com/USEPA/WNTR/blob/main/documentation/units.rst>
- **Validated quotes:** `LPS: L/s`; `Pressure head = ... m`.
- **Supported claim:** In the EPANET metric convention, `LPS` denotes liters per
  second and pressure head is measured in meters.

The exact quotations above were checked against downloaded source text with the
research skill's `validate_quote.py` utility. An adversarial review required the
claim restrictions below before the sources could inform implementation.

## Local Dataset Evidence

A local aggregate over only `train.iloc[0:5840]` produced:

- 5,840 rows, 43 numeric channels, no missing or non-finite cells.
- Prefix counts: 7 `L_`, 12 `F_`, 12 `S_`, and 12 `P_` channels.
- All observed `S_*` values in this block are exactly `0.0` or `1.0`.
- Seven channels are constant in this block: `S_PU1`, `F_PU3`, `S_PU3`,
  `F_PU5`, `S_PU5`, `F_PU9`, and `S_PU9`.
- `L_T1` ranges from `0.32011184096336365` to `4.871329307556152`; its maximum
  observed absolute one-step change is `0.6543217897415161`.

## Permitted Inferences and Limits

1. Prefix-based element roles and EPANET units are candidate public-model
   semantics, not yet verified BATADAL stored-data semantics.
2. Public identifiers may be recorded as candidate topology mappings, but
   `dataset_alignment_validated` remains false until the exact generator/model
   relationship is verified and sealed.
3. Stored units remain unknown. Candidate units may be recorded separately;
   they cannot justify hard constraints or unit conversion.
4. `S_*` may be classified as observed binary in the model-fit block because
   its values were profiled, not because the prefix proves binary behavior.
5. Non-target channels are `declared_unaffected` only under the synthetic
   benchmark threat model; this is not independent physical trust.

## Implementation Consequence

The first manifest must expose semantic validation status, stored-unit status,
constraint enforcement, evidence scope, evidence sources, and topology alignment
explicitly. All current empirical and public-physics constraints are soft. No
generic nonnegative-flow constraint is allowed because link-flow sign can depend
on orientation.
