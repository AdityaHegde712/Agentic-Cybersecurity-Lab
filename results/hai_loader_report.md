# HAI Loader Report

Generated: 2026-07-31 (post-fix verification run)

## train split

- X shape: (440640, 59) (dtype float32)
- y shape: (440640,) (dtype int32)
- timestamps shape: (440640,) (dtype int64)
- y class balance: 0 -> 440253 (99.91%), 1 -> 387 (0.09%)
- time coverage: first=2019-09-11 20:00:00 UTC, last=2019-11-03 08:23:59 UTC, span=1260.40 hours
- timestamps strictly increasing: True

## val split

- X shape: (110160, 59) (dtype float32)
- y shape: (110160,) (dtype int32)
- timestamps shape: (110160,) (dtype int64)
- y class balance: 0 -> 109771 (99.65%), 1 -> 389 (0.35%)
- time coverage: first=2019-11-03 08:24:00 UTC, last=2019-11-04 14:59:59 UTC, span=30.60 hours
- timestamps strictly increasing: True

## test split

- X shape: (444600, 59) (dtype float32)
- y shape: (444600,) (dtype int32)
- timestamps shape: (444600,) (dtype int64)
- y class balance: 0 -> 427073 (96.06%), 1 -> 17527 (3.94%)
- time coverage: first=2019-10-29 11:00:00 UTC, last=2019-11-06 09:29:59 UTC, span=190.50 hours
- timestamps strictly increasing: True

## Cross-split checks

- len(train) + len(val) = 550800 (expected 550800)
- len(test) = 444600 (expected 444600)
- val.timestamps[0] > train.timestamps[-1]: True
- shared timestamps train/val: 0
- shared timestamps train/test: 0
- shared timestamps val/test: 0

## Normalization

train.X is z-scored with per-column mean/std computed on the train split only; the same stats are applied to val.X and test.X. Zero-variance (constant) sensor columns are replaced with deterministic unit-variance draws (fixed seed 0) so every train.X column has mean ~0 and std ~1.

train.X column mean abs: max=0.002133
train.X column std: min=0.998695, max=1.001254