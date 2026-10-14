"""1D Total-Variation / Fused Lasso (MAP) Step Estimator for sensor additive attacks.

Solves the deterministic optimization:
    min_delta  0.5 * ||y - x_hat_surrogate - delta||_2^2
               + lambda_tv * ||D delta||_1
               + lambda_l1 * ||delta||_1
subject to bounds:
    x_min <= y - delta <= x_max   <=>   y - x_max <= delta <= y - x_min
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


class StepMAPEstimator(BaseDeltaEstimator):
    """1D Total-Variation and L1 Fused Lasso MAP Step Estimator under physical bounds."""

    def __init__(
        self,
        lambda_tv: float = 1.0,
        lambda_l1: float = 0.1,
        target_channel: str = "L_T1",
        bounds: tuple[float, float] | None = None,
        channel_manifest_path: str | Path | None = None,
        surrogate_estimator: BaseDeltaEstimator | None = None,
        solver: SolverType = "admm",
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        if lambda_tv < 0.0:
            raise ValueError("lambda_tv must be non-negative")
        if lambda_l1 < 0.0:
            raise ValueError("lambda_l1 must be non-negative")
        if solver not in ("admm", "scipy"):
            raise ValueError("solver must be either 'admm' or 'scipy'")

        self.lambda_tv = float(lambda_tv)
        self.lambda_l1 = float(lambda_l1)
        self.target_channel = target_channel
        self.surrogate_estimator = surrogate_estimator
        self.solver: SolverType = solver
        self.max_iter = max_iter
        self.tol = float(tol)

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

    def fit_normal(
        self,
        normal_data: pd.DataFrame | np.ndarray,
        **kwargs: Any,
    ) -> StepMAPEstimator:
        """Fit underlying surrogate estimator if present."""
        if self.surrogate_estimator is not None:
            self.surrogate_estimator.fit_normal(normal_data, **kwargs)
        return self

    def reconstruct(
        self,
        observed: np.ndarray | pd.Series | pd.DataFrame,
        surrogate_x: np.ndarray | None = None,
        attack_interval: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> ReconstructionResult:
        """Reconstruct clean signal and step attack trajectory via MAP optimization."""
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

        # 2. Extract surrogate clean prediction
        if surrogate_x is not None:
            x_surr = np.asarray(surrogate_x, dtype=np.float64)
            if len(x_surr) != n:
                raise ValueError(f"surrogate_x length ({len(x_surr)}) must match observed length ({n})")
        elif self.surrogate_estimator is not None:
            surr_res = self.surrogate_estimator.reconstruct(observed, **kwargs)
            x_surr = surr_res.x_hat
        else:
            x_surr = y.copy()

        # 3. Determine box bounds on delta: y - x_max <= delta <= y - x_min
        x_min, x_max = self.bounds
        lower = np.full(n, -np.inf, dtype=np.float64)
        upper = np.full(n, np.inf, dtype=np.float64)

        if np.isfinite(x_max):
            lower = y - x_max
        if np.isfinite(x_min):
            upper = y - x_min

        # Enforce declared attack interval (off-support delta = 0)
        if attack_interval is not None:
            start, stop = attack_interval
            if not (0 <= start <= stop <= n):
                raise ValueError(f"Invalid attack_interval {attack_interval} for length {n}")
            lower[:start] = 0.0
            upper[:start] = 0.0
            lower[stop:] = 0.0
            upper[stop:] = 0.0

        # Ensure lower <= upper
        lower = np.minimum(lower, upper)

        # 4. Solve optimization
        residual = y - x_surr

        if self.solver == "admm":
            delta_hat, iters, loss = self._solve_admm(residual, lower, upper)
        else:
            delta_hat, iters, loss = self._solve_scipy(residual, lower, upper)

        # Final safety projection onto bounds
        delta_hat = np.clip(delta_hat, lower, upper)
        if attack_interval is not None:
            start, stop = attack_interval
            delta_hat[:start] = 0.0
            delta_hat[stop:] = 0.0

        x_hat = y - delta_hat
        exact_loss = self.compute_objective(residual, delta_hat)

        prov = {
            "estimator": self.__class__.__name__,
            "solver": self.solver,
            "lambda_tv": self.lambda_tv,
            "lambda_l1": self.lambda_l1,
            "bounds": self.bounds,
            "iterations": iters,
            "objective_loss": exact_loss,
        }

        return ReconstructionResult(
            x_hat=x_hat,
            delta_hat=delta_hat,
            pointwise_uncertainty=np.zeros(n, dtype=np.float64),
            pointwise_variance=np.zeros(n, dtype=np.float64),
            objective_loss=exact_loss,
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

    def _solve_admm(
        self,
        residual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        rho: float = 2.0,
    ) -> tuple[np.ndarray, int, float]:
        """Solve bounded TV + L1 MAP via ADMM proximal splitting."""
        n = len(residual)
        if n == 1:
            best_d = float(np.clip(residual[0], lower[0], upper[0]))
            return np.array([best_d]), 1, self.compute_objective(residual, np.array([best_d]))

        # Linear system banded representation for (1 + 2*rho)*I + rho * D^T D
        # D^T D is tridiagonal:
        # main diag: 1 at index 0, 2 at index 1..n-2, 1 at index n-1
        diag = np.full(n, 1.0 + 2.0 * rho, dtype=np.float64)
        diag[0] += rho * 1.0
        diag[1:-1] += rho * 2.0
        diag[-1] += rho * 1.0
        off_diag = np.full(n - 1, -rho, dtype=np.float64)

        ab = np.zeros((3, n), dtype=np.float64)
        ab[0, 1:] = off_diag
        ab[1, :] = diag
        ab[2, :-1] = off_diag

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
            # 1. Delta update: solve banded tridiagonal system
            v2 = z2 - u2
            dt_v2 = np.zeros(n, dtype=np.float64)
            dt_v2[0] = -v2[0]
            if n > 2:
                dt_v2[1:-1] = v2[:-1] - v2[1:]
            dt_v2[-1] = v2[-1]

            rhs = residual + rho * (z0 - u0) + rho * (z1 - u1) + rho * dt_v2
            delta = la.solve_banded((1, 1), ab, rhs)

            # 2. Auxiliary updates
            z0_old = z0
            z0 = np.clip(delta + u0, lower, upper)
            z1 = _soft_threshold(delta + u1, self.lambda_l1 / rho)
            d_delta = np.diff(delta)
            z2 = _soft_threshold(d_delta + u2, self.lambda_tv / rho)

            # 3. Dual updates
            u0 += delta - z0
            u1 += delta - z1
            u2 += d_delta - z2

            # Check convergence
            prim_res = max(
                np.max(np.abs(delta - z0)),
                np.max(np.abs(delta - z1)),
                np.max(np.abs(d_delta - z2)),
            )
            dual_res = np.max(np.abs(z0 - z0_old))
            if prim_res < self.tol and dual_res < self.tol and it > 5:
                break

        loss = self.compute_objective(residual, delta)
        return delta, iter_count, loss

    def _solve_scipy(
        self,
        residual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, int, float]:
        """Solve bounded TV + L1 MAP via Scipy L-BFGS-B with smooth Huber approximation."""
        n = len(residual)
        bounds = list(zip(lower, upper))
        x0 = np.clip(residual, lower, upper)

        def objective(d: np.ndarray) -> float:
            data_term = 0.5 * np.sum((residual - d) ** 2)
            l1_term = self.lambda_l1 * np.sum(np.sqrt(d ** 2 + eps ** 2))
            if n > 1:
                diffs = np.diff(d)
                tv_term = self.lambda_tv * np.sum(np.sqrt(diffs ** 2 + eps ** 2))
            else:
                tv_term = 0.0
            return float(data_term + l1_term + tv_term)

        def jacobian(d: np.ndarray) -> np.ndarray:
            g = -(residual - d)
            if self.lambda_l1 > 0:
                g += self.lambda_l1 * (d / np.sqrt(d ** 2 + eps ** 2))
            if n > 1 and self.lambda_tv > 0:
                diffs = np.diff(d)
                d_term = self.lambda_tv * (diffs / np.sqrt(diffs ** 2 + eps ** 2))
                g[:-1] -= d_term
                g[1:] += d_term
            return g

        res = minimize(
            objective,
            x0,
            jac=jacobian,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )

        loss = self.compute_objective(residual, res.x)
        return res.x, res.nit, loss
