# Narrow Additive Inverse Identifiability and Noise Contract

**Status:** proposed implementation contract pending executable-test and
adversarial-review gates.  
**Benchmark:** BATADAL `L_T1`, known target sensor, known attack interval,
bounded signed finite-duration step.

## Claim Boundary

The project estimates `delta_x` only conditional on a declared normal-dynamics
class, attack class, data split, trusted-input threat model, and constraint
manifest. It does not claim that `x` and `delta_x` are uniquely recoverable from
`y` alone.

## Observation Models

### Synthetic Injection Evaluation

For a recorded clean held-out sequence,

`y_t = x_recorded,t + delta_t`

is exact by construction. Normal physical variation and any measurement noise
already present in the clean record belong to `x_recorded`; the injected
`delta` is stored directly and is the only trajectory-level recovery ground
truth. Numerical equality is evaluated with the generator's declared floating-
point contract, not inferred from nonzero residuals.

### Real Attack Case Study

For a physical or simulated attack record without its clean counterfactual,

`z_(t+1) = f_theta(z_t, u_t) + w_t`

`q_t = h_theta(z_t, u_t)`

`x_t = q_t + epsilon_t`

`y_t = x_t + delta_t`.

Here `q_t` is the latent physical measurement, `x_t` is the clean sensor reading,
`w_t` is process/model discrepancy, and `epsilon_t` is observation noise. Their
distributions are fitted or calibrated only on authorized normal blocks. The
project's additive identity is always defined against `x_t`, never directly
against `q_t`. Inference may target `delta_t` conditional on the latent model or
the counterfactual clean reading `x_t`, but the target definition must be sealed
before evaluation. Real attack labels can identify a reported interval or
affected component, but they do not provide trajectory ground truth for either
`x_t` or `delta_t`. Real-data conclusions are therefore posterior estimates and
qualitative consistency checks; trajectory recovery error is prohibited without
a recorded counterfactual.

## Hard Feasible Set and Probabilistic Set

For the first benchmark, the target channel `j` and interval `W` are supplied.
A candidate pair `(x, delta)` belongs to the deterministic set `F_hard(y)` only
if:

1. `y = x + delta` satisfies the declared numerical equality tolerance.
2. `delta_(t,k) = 0` for `t` outside `W` or `k != j` under the synthetic threat
   model.
3. `delta` belongs to the sealed attack class, including explicit amplitude
   interval, sign set, duration interval, and change-point convention.
4. `x` belongs to an explicitly parameterized normal-dynamics class fitted
   without attack or final-block leakage.
5. Every hard constraint has validated provenance and its numeric parameters are
   frozen in the benchmark seal.

Let `Delta_hard(y)` be the projection of `F_hard(y)` onto `delta`. For a sealed
norm `||.||_D` and tolerance `tau_delta >= 0`, practical structural
identifiability requires a global certificate that

`diam_D(Delta_hard(y)) = sup_(d1,d2 in Delta_hard(y)) ||d1-d2||_D <= tau_delta`.

The certificate method must be sealed: either exhaustive enumeration for a
finite class or certified global extrema/bounds for a continuous class. A finite
candidate search or one optimizer solution cannot certify this condition.
The executable certificate carries the method, exact `F_hard(y)` coverage scope,
a canonical JSON evidence manifest and its verified SHA-256, the nonempty set of
verified hard-constraint IDs, and a finite nonnegative certified upper bound on
`diam_D(Delta_hard(y))`. It is bound to a nonblank benchmark-case ID, a
deterministic fingerprint of the exact observed vector, the sealed delta norm,
and canonical JSON plus verified SHA-256 for the full hard-set manifest. The
evidence-manifest fields must exactly match the typed certificate and hard-set
seal. `unique_certified` is permitted only when every binding matches the active
assessment and the bound is at most the sealed `tau_delta`; a larger upper bound
is inconclusive. A caller boolean, a SHA-shaped string without its resolved
payload, or an exhaustive scan of only a supplied candidate list is not a
hard-set certificate. Any supplied feasible witness whose separation exceeds
the certificate's upper bound invalidates the certificate.

The hard-set manifest is the definition of `F_hard(y)`, not an ID list. Its
canonical payload seals the exact support mask/window, numerical identity
tolerance, attack-class amplitude/sign/duration/change-point convention,
normal-dynamics artifact and SHA-256, and every hard constraint with its typed
numeric parameters. A certified assessment must receive that active payload,
verify its digest, and match its support and identity tolerance to the function
inputs. Changing any of those fields creates a different hard set and invalidates
certificate reuse.

A probabilistic model instead yields a calibrated confidence or credible set
`C_(1-alpha)(y)`. Its level `1-alpha`, construction, likelihood, priors, and
calibration data must be frozen before final evaluation. Diameter, mass, and
modes of this set are posterior-concentration diagnostics, not structural
identifiability. An unthresholded Gaussian likelihood is never used as the hard
feasible set merely because every finite residual has nonzero density.

## Required Toy Strengthened-Screen Counterexample

The following one-channel example uses the same observation and known support
`W = {2, 3}`:

| Quantity | t0 | t1 | t2 | t3 | t4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `y` | 1.0 | 1.0 | 2.0 | 2.0 | 1.0 |
| `x_A` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `delta_A` | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| `x_B` | 1.0 | 1.0 | 1.5 | 1.5 | 1.0 |
| `delta_B` | 0.0 | 0.0 | 0.5 | 0.5 | 0.0 |

This example deliberately strengthens the current soft `L_T1` screen into an
illustrative hard normal class: every clean value is inside
`[0.32011184096336365, 4.871329307556152]` and every clean one-step absolute
change is at most `0.6543217897415161`. Its illustrative attack-amplitude set
contains `0.5` and `1.0`, with duration two and support exactly `W`.

Both decompositions satisfy that toy class, exact addition, and the same support.
More generally, set `x_t = 1+a` and `delta_t = 1-a` on `W`, leaving `x_t=1` and
`delta_t=0` elsewhere. Every `a` in `[0, 0.6543217897415161]` meets the toy
range/rate screen, giving a continuum of attack amplitudes
`1-a`. Therefore empirical bounds and rates would not identify attack amplitude
even if promoted to hard restrictions. This counterexample does not prove
ambiguity under the eventual fitted BATADAL dynamics class or a different sealed
attack-amplitude set.

## Required Ambiguity Response

The benchmark seals `tau_delta`, `||.||_D`, and a certificate method before the
final block. The estimator reports exactly one status:

1. `ambiguous` when a constructive pair or certified diameter exceeds
   `tau_delta` and both members have verified membership in the same sealed
   `F_hard(y)`;
2. `unique_certified` only when a global certificate proves the hard-set
   diameter is at most `tau_delta`; or
3. `not_certified` when the search is incomplete and no constructive ambiguous
   pair has been found.

Two supplied candidates that pass only addition and support checks are a useful
`candidate_ambiguity_unverified` diagnostic, but they do not establish structural
ambiguity. Until a feasibility witness verifies both candidates against the same
sealed attack class, normal dynamics, and hard constraints, structural status
remains `not_certified`.

Every report includes the norm, `tau_delta`, maximum witnessed or certified
separation, certificate/search method, and candidate-separating assumptions
actually evaluated. If none were evaluated, it states `not identified`.
A posterior report separately records credible-set level and construction,
diameter or coordinatewise intervals, and mode/mass diagnostics. It never maps a
single MAP solution or a sampled absence of a second mode to `unique_certified`.

The low-level candidate-separation helper retains a backward-compatible
`certified` field meaning only that the supplied candidate collection was
declared exhaustively enumerated. Its `certification_scope` must say
`supplied_candidate_enumeration`, and it can never produce structural
`unique_certified` without the evidence-backed hard-set certificate above.

In synthetic evaluation, ambiguity is reported alongside error against the
known injected delta. In a real attack case study, ambiguity prevents a unique
trajectory claim because no counterfactual delta label exists.

## Trust and Constraint Rules

- `declared_unaffected` is an oracle assumption of the synthetic threat model,
  not proof of physical cleanliness, calibration, or secure communication.
- Outside synthetic evaluation, `declared_unaffected` never satisfies the
  independently trusted-input requirement by itself.
- Every tunable attack prior, model family, ambiguity norm/tolerance, likelihood,
  calibration method, and manual diagnostic belongs to a named fit, selection,
  calibration, or development-injection block and is hashed before final use.
- A causal estimate at time `t` may use only `{y_<=t, u_<=t, independently
  trusted companions_<=t}` and recursively reconstructed target history. Future
  values are reserved for a separately labeled offline smoother.
- A future hard physical rule requires validated topology alignment, validated
  stored units and semantics, rule-specific validation on authorized normal
  data, and an explicit trust/threat-model designation for every input. Current
  public-model physics and all empirical constraints remain soft and ablated.
- Empirical bounds and rates use only model-fit rows `0:5840`; they cannot be
  recalibrated on development or final injections.
- Clean calibration may estimate likelihood/noise parameters only after the
  dynamics form is frozen; the sealed final block cannot tune any component.

## Prohibited Claims

The study must not claim general observation-only inversion, direct real-attack
delta accuracy, trusted physics from correlation, causal recovery from an
offline smoother, or uniqueness when the feasible-set/posterior ambiguity gate
fails.

## Task 4 Exit Criteria

Task 4 is complete only when the observation-regime semantics and ambiguity
witness are locked by executable tests, the witness is reproduced locally, an
adversarial review accepts the claim boundary, and the accepted contract is
recorded in `DECISIONS.md` and the sprint execution report.
