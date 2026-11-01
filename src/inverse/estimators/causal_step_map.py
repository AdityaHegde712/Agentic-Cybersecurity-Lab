"""Strictly Causal Rolling 1D Total-Variation (MAP) Step Estimator.

Phase 3 Milestone: Past-Only Causal Rolling Filter.

Estimates the attack trajectory delta_t and clean state x_t strictly forward in time
using a sliding lookback window [max(0, t - window_size + 1) : t + 1], without conditioning
on or observing future samples y_{t+1:N}.

Solves at each timestep t:
    min_delta  0.5 * ||(y - x_hat_surrogate)_w - delta_w||_2^2
               + lambda_tv * ||D delta_w||_1
               + lambda_l1 * ||delta_w||_1
subject to physical bounds:
    x_min <= y_w - delta_w <= x_max   <=>   y_w - x_max <= delta_w <= y_w - x_min
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import minimize

from src.inverse.channel_constraints import ChannelConstraintManifest
from src.inverse.estimators.base import BaseDeltaEstimator, ReconstructionResult


SolverType = Literal["admm", "scipy"]


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Soft thresholding operator: sign(x) * max(|x| - threshold, 0)."""
    if threshold <= 0.0:
        return x
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


class CausalStepMAPEstimator(BaseDeltaEstimator):
    """Past-Only Causal Rolling Filter for Additive Step Attack Recovery."""

    def __init__(
        self,
        window_size: int = 48,
        lambda_tv: float = 1.0,
        lambda_l1: float = 0.1,
        target_channel: str = "L_T1",
        bounds: tuple[float, float] | None = None,
        channel_manifest_path: str | Path | None = None,
        surrogate_estimator: BaseDeltaEstimator | Any | None = None,
        residual_std: float | None = None,
        solver: SolverType = "admm",
        max_iter: int = 300,
        tol: float = 1e-5,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if lambda_tv < 0.0:
            raise ValueError("lambda_tv must be non-negative")
        if lambda_l1 < 0.0:
            raise ValueError("lambda_l1 must be non-negative")
        if solver not in ("admm", "scipy"):
            raise ValueError("solver must be either 'admm' or 'scipy'")

        self.window_size = int(window_size)
        self.lambda_tv = float(lambda_tv)
        self.lambda_l1 = float(lambda_l1)
        self.target_channel = target_channel
        self.surrogate_estimator = surrogate_estimator
        self._residual_std = float(residual_std) if residual_std is not None else None
        self.solver: SolverType = solver
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Cholesky factorization cache for ADMM tridiagonal system: {window_len: (cb, lower)}
        self._cho_cache: dict[int, tuple[np.ndarray, bool]] = {}

        if bounds is not None:
            if bounds[0] > bounds[1]:
                raise ValueError("bounds minimum must not exceed maximum")
            self.bounds = (float(bounds[0]), float(bounds[1]))
        elif channel_manifest_path is not None:
            manifest_file = Path(channel_manifest_path)
            if not manifest_file.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_file}")
            manifest = ChannelConstraintManifest.from_json(manifest_file.read_text(encoding="utf-8"))
            ch = next((c for c in manifest.channels if c.name == target_channel), None)
            if ch is None:
                raise ValueError(f"Target channel {target_channel} not found in manifest")
            self.bounds = (ch.empirical_bounds.minimum, ch.empirical_bounds.maximum)
        else:
            default_path = Path("configs/manifests/batadal-channel-manifest-v1.json")
            if default_path.exists():
                manifest = ChannelConstraintManifest.from_json(default_path.read_text(encoding="utf-8"))
                ch = next((c for c in manifest.channels if c.name == target_channel), None)
                if ch is not None:
                    self.bounds = (ch.empirical_bounds.minimum, ch.empirical_bounds.maximum)
                else:
                    self.bounds = (-np.inf, np.inf)
            else:
                self.bounds = (-np.inf, np.inf)

    @property
    def residual_std_(self) -> float:
        """Calibrated surrogate residual standard deviation."""
        if self._residual_std is not None and self._residual_std > 0.0:
            return self._residual_std
        if self.surrogate_estimator is not None:
            if hasattr(self.surrogate_estimator, "residual_std_") and self.surrogate_estimator.residual_std_ > 0.0:
                return float(self.surrogate_estimator.residual_std_)
            if hasattr(self.surrogate_estimator, "residual_variance") and self.surrogate_estimator.residual_variance > 0.0:
                return float(np.sqrt(self.surrogate_estimator.residual_variance))
            if hasattr(self.surrogate_estimator, "residual_variance_") and self.surrogate_estimator.residual_variance_ > 0.0:
                return float(np.sqrt(self.surrogate_estimator.residual_variance_))
        return 0.0

    @residual_std_.setter
    def residual_std_(self, value: float) -> None:
        self._residual_std = float(value)

    def fit_normal(
        self,
        normal_data: pd.DataFrame | np.ndarray,
        **kwargs: Any,
    ) -> CausalStepMAPEstimator:
        """Fit underlying surrogate estimator if present and calibrate residual std."""
        if self.surrogate_estimator is not None:
            if hasattr(self.surrogate_estimator, "fit_normal"):
                self.surrogate_estimator.fit_normal(normal_data, **kwargs)
            elif hasattr(self.surrogate_estimator, "fit"):
                if not getattr(self.surrogate_estimator, "is_fitted", False):
                    if isinstance(normal_data, pd.DataFrame):
                        self.surrogate_estimator.fit(normal_data, **kwargs)
            if hasattr(self.surrogate_estimator, "residual_std_") and self.surrogate_estimator.residual_std_ > 0.0:
                self._residual_std = float(self.surrogate_estimator.residual_std_)
            elif hasattr(self.surrogate_estimator, "residual_variance") and self.surrogate_estimator.residual_variance > 0.0:
                self._residual_std = float(np.sqrt(self.surrogate_estimator.residual_variance))
            elif hasattr(self.surrogate_estimator, "residual_variance_") and self.surrogate_estimator.residual_variance_ > 0.0:
                self._residual_std = float(np.sqrt(self.surrogate_estimator.residual_variance_))
        return self

    def reconstruct(
        self,
        observed: np.ndarray | pd.Series | pd.DataFrame,
        surrogate_x: np.ndarray | None = None,
        attack_interval: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> ReconstructionResult:
        """Reconstruct clean signal and attack trajectory online forward in time."""
        # 1. Extract observed 1D vector y
        if isinstance(observed, pd.DataFrame):
            if self.target_channel not in observed.columns:
                raise ValueError(f"Target channel {self.target_channel} not in DataFrame")
            y = observed[self.target_channel].to_numpy(dtype=np.float64)
        elif isinstance(observed, pd.Series):
            y = observed.to_numpy(dtype=np.float64)
        else:
            y = np.asarray(observed, dtype=np.float64)

        n = len(y)
        if n == 0:
            return ReconstructionResult(
                x_hat=np.array([], dtype=np.float64),
                delta_hat=np.array([], dtype=np.float64),
                pointwise_uncertainty=np.array([], dtype=np.float64),
                pointwise_variance=np.array([], dtype=np.float64),
                objective_loss=0.0,
                ambiguity_flag=False,
                provenance={"estimator": self.__class__.__name__, "causal": True},
            )

        # 2. Extract surrogate clean prediction and uncertainty scale
        surr_unc = self.residual_std_
        if surrogate_x is not None:
            x_surr = np.asarray(surrogate_x, dtype=np.float64)
            if len(x_surr) != n:
                raise ValueError(f"surrogate_x length ({len(x_surr)}) must match observed length ({n})")
        elif self.surrogate_estimator is not None:
            surr_res = self.surrogate_estimator.reconstruct(observed, **kwargs)
            x_surr = surr_res.x_hat
            if surr_unc == 0.0 and surr_res.pointwise_uncertainty is not None and len(surr_res.pointwise_uncertainty) > 0:
                surr_unc = float(np.mean(surr_res.pointwise_uncertainty))
        else:
            x_surr = y.copy()

        if surr_unc == 0.0 and "residual_std" in kwargs:
            surr_unc = float(kwargs["residual_std"])

        full_residuals = y - x_surr
        x_min, x_max = self.bounds

        delta_hat = np.zeros(n, dtype=np.float64)
        total_loss = 0.0
        prev_solution: np.ndarray | None = None

        # 3. Strictly Causal Rolling Optimization Loop
        for t in range(n):
            w_start = max(0, t - self.window_size + 1)
            w_end = t + 1
            w_len = w_end - w_start

            # Past-only window of observations and residuals
            r_w = full_residuals[w_start:w_end]
            y_w = y[w_start:w_end]

            # Box bounds: y - x_max <= delta <= y - x_min
            lower_w = np.full(w_len, -np.inf, dtype=np.float64)
            upper_w = np.full(w_len, np.inf, dtype=np.float64)
            if np.isfinite(x_max):
                lower_w = y_w - x_max
            if np.isfinite(x_min):
                upper_w = y_w - x_min

            # Oracle attack interval constraint if provided
            if attack_interval is not None:
                onset, offset = attack_interval
                window_indices = np.arange(w_start, w_end)
                off_support_mask = (window_indices < onset) | (window_indices >= offset)
                lower_w[off_support_mask] = 0.0
                upper_w[off_support_mask] = 0.0

            lower_w = np.minimum(lower_w, upper_w)

            # Solve 1D MAP on rolling window
            if w_len == 1:
                sol = np.clip(r_w, lower_w, upper_w)
                step_loss = self.compute_objective(r_w, sol)
            elif self.solver == "admm":
                # Prepare warm start from previous overlapping window
                warm_delta = None
                if prev_solution is not None and len(prev_solution) > 0:
                    warm_delta = np.zeros(w_len, dtype=np.float64)
                    if len(prev_solution) == w_len:
                        warm_delta[:-1] = prev_solution[1:]
                        warm_delta[-1] = prev_solution[-1]
                    elif len(prev_solution) < w_len:
                        warm_delta[: len(prev_solution)] = prev_solution
                        warm_delta[len(prev_solution) :] = prev_solution[-1]

                sol, _, step_loss = self._solve_admm(r_w, lower_w, upper_w, warm_delta=warm_delta)
            else:
                sol, _, step_loss = self._solve_scipy(r_w, lower_w, upper_w)

            sol = np.clip(sol, lower_w, upper_w)
            prev_solution = sol

            # Filter commitment: assign estimate for current timestep t (last element of window)
            delta_hat[t] = sol[-1]
            total_loss += step_loss

        # Ensure off-support zeros if attack_interval is declared
        if attack_interval is not None:
            onset, offset = attack_interval
            delta_hat[:onset] = 0.0
            delta_hat[offset:] = 0.0

        x_hat = y - delta_hat
        unc_arr = np.full(n, surr_unc, dtype=np.float64)

        prov = {
            "estimator": self.__class__.__name__,
            "solver": self.solver,
            "lambda_tv": self.lambda_tv,
            "lambda_l1": self.lambda_l1,
            "window_size": self.window_size,
            "causal": True,
            "bounds": self.bounds,
            "objective_loss": float(total_loss / n if n > 0 else 0.0),
        }

        return ReconstructionResult(
            x_hat=x_hat,
            delta_hat=delta_hat,
            pointwise_uncertainty=unc_arr,
            pointwise_variance=unc_arr**2,
            objective_loss=float(total_loss / n if n > 0 else 0.0),
            ambiguity_flag=False,
            provenance=prov,
        )

    def compute_objective(self, residual: np.ndarray, delta: np.ndarray) -> float:
        """Compute the deterministic scalar MAP objective."""
        data_fidelity = 0.5 * np.sum((residual - delta) ** 2)
        l1_penalty = self.lambda_l1 * np.sum(np.abs(delta))
        if len(delta) > 1:
            tv_penalty = self.lambda_tv * np.sum(np.abs(np.diff(delta)))
        else:
            tv_penalty = 0.0
        return float(data_fidelity + tv_penalty + l1_penalty)

    def _get_cholesky_factors(self, n: int, rho: float) -> tuple[np.ndarray, bool]:
        """Obtain or compute cached Cholesky factor for (1 + 2*rho)*I + rho*D^T D."""
        if n in self._cho_cache:
            return self._cho_cache[n]

        diag = np.full(n, 1.0 + 2.0 * rho, dtype=np.float64)
        diag[0] += rho * 1.0
        diag[1:-1] += rho * 2.0
        diag[-1] += rho * 1.0
        off_diag = np.full(n - 1, -rho, dtype=np.float64)

        ab_sym = np.zeros((2, n), dtype=np.float64)
        ab_sym[0, :] = diag
        ab_sym[1, :-1] = off_diag

        cb = la.cholesky_banded(ab_sym, lower=True)
        factors = (cb, True)
        self._cho_cache[n] = factors
        return factors

    def _solve_admm(
        self,
        residual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        rho: float = 2.0,
        warm_delta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Solve bounded TV + L1 MAP via ADMM proximal splitting with Cholesky solves."""
        n = len(residual)
        if n == 1:
            best_d = float(np.clip(residual[0], lower[0], upper[0]))
            return np.array([best_d]), 1, self.compute_objective(residual, np.array([best_d]))

        cho_factors = self._get_cholesky_factors(n, rho)

        if warm_delta is not None and len(warm_delta) == n:
            delta = np.clip(warm_delta, lower, upper)
        else:
            delta = np.clip(residual, lower, upper)

        z0 = delta.copy()
        z1 = delta.copy()
        z2 = np.diff(delta)

        u0 = np.zeros(n, dtype=np.float64)
        u1 = np.zeros(n, dtype=np.float64)
        u2 = np.zeros(n - 1, dtype=np.float64)

        iter_count = 0
        for it in range(self.max_iter):
            iter_count = it + 1
            # 1. Delta update: solve banded tridiagonal system via cached Cholesky
            v2 = z2 - u2
            dt_v2 = np.zeros(n, dtype=np.float64)
            dt_v2[0] = -v2[0]
            if n > 2:
                dt_v2[1:-1] = v2[:-1] - v2[1:]
            dt_v2[-1] = v2[-1]

            rhs = residual + rho * (z0 - u0) + rho * (z1 - u1) + rho * dt_v2
            delta_new = la.cho_solve_banded(cho_factors, rhs)

            # 2. Auxiliary variable updates (proximal operators)
            # z0: box projection
            z0 = np.clip(delta_new + u0, lower, upper)

            # z1: soft-thresholding for L1 sparsity
            z1 = _soft_threshold(delta_new + u1, self.lambda_l1 / rho)

            # z2: soft-thresholding for TV
            d_delta = np.diff(delta_new)
            z2 = _soft_threshold(d_delta + u2, self.lambda_tv / rho)

            # 3. Dual updates
            u0 += delta_new - z0
            u1 += delta_new - z1
            u2 += d_delta - z2

            # 4. Check convergence
            primal_res = max(
                np.max(np.abs(delta_new - z0)),
                np.max(np.abs(delta_new - z1)),
                np.max(np.abs(d_delta - z2)),
            )
            step_change = np.max(np.abs(delta_new - delta))
            delta = delta_new

            if primal_res < self.tol and step_change < self.tol:
                break

        loss = self.compute_objective(residual, delta)
        return delta, iter_count, loss

    def _solve_scipy(
        self,
        residual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        """Solve bounded TV + L1 MAP via Scipy L-BFGS-B optimization."""
        n = len(residual)
        bounds = [(lower[i], upper[i]) for i in range(n)]
        x0 = np.clip(residual, lower, upper)

        def obj(d: np.ndarray) -> float:
            return self.compute_objective(residual, d)

        res = minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        delta_hat = res.x
        return delta_hat, int(res.nit), float(res.fun)
