"""Multi-Order Trend Filtering MAP Estimators for Sensor Attack Recovery.

Sprint 4B: Attack Family Generalization via Multi-Order Trend Filtering.

Formulation:
    min_delta  0.5 * w_eps * ||y - x_hat_surrogate - delta||_2^2
               + lambda_0 * ||delta||_1
               + lambda_1 * ||D^1 delta||_1
               + lambda_2 * ||D^2 delta||_1
subject to physical bounds:
    x_min <= y - delta <= x_max   <=>   y - x_max <= delta <= y - x_min

where:
- w_eps = 1.0 / (sigma_epsilon ** 2)
- D^1 is the first-order difference operator (inducing piecewise-constant profiles)
- D^2 is the second-order difference operator (inducing piecewise-linear profiles with sparse kinks)
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


def _build_pentadiagonal_cholesky(
    n: int,
    w_eps: float,
    rho_0: float,
    rho_1: float,
    rho_2: float,
) -> tuple[np.ndarray, bool]:
    """Build and factorize symmetric positive-definite pentadiagonal system via Cholesky."""
    if n == 2:
        ab = np.zeros((3, 2), dtype=np.float64)
        diag = np.full(2, w_eps + rho_0 + rho_1, dtype=np.float64)
        ab[0, :] = diag
        ab[1, 0] = -rho_1
        cb = la.cholesky_banded(ab, lower=True)
        return cb, True

    ab = np.zeros((3, n), dtype=np.float64)

    # 1. Main diagonal
    diag = np.full(n, w_eps + rho_0, dtype=np.float64)
    if rho_1 > 0.0:
        diag[0] += rho_1
        diag[1:-1] += 2.0 * rho_1
        diag[-1] += rho_1
    if rho_2 > 0.0:
        if n == 3:
            diag[0] += rho_2
            diag[1] += 4.0 * rho_2
            diag[2] += rho_2
        elif n == 4:
            diag[0] += rho_2
            diag[1] += 5.0 * rho_2
            diag[2] += 5.0 * rho_2
            diag[3] += rho_2
        else:
            diag[0] += rho_2
            diag[1] += 5.0 * rho_2
            diag[2:-2] += 6.0 * rho_2
            diag[-2] += 5.0 * rho_2
            diag[-1] += rho_2
    ab[0, :] = diag

    # 2. First subdiagonal
    sub1 = np.zeros(n - 1, dtype=np.float64)
    if rho_1 > 0.0:
        sub1 -= rho_1
    if rho_2 > 0.0:
        if n == 3:
            sub1 += np.array([-2.0 * rho_2, -2.0 * rho_2], dtype=np.float64)
        elif n == 4:
            sub1 += np.array([-2.0 * rho_2, -4.0 * rho_2, -2.0 * rho_2], dtype=np.float64)
        else:
            sub1[0] -= 2.0 * rho_2
            sub1[-1] -= 2.0 * rho_2
            sub1[1:-1] -= 4.0 * rho_2
    ab[1, :-1] = sub1

    # 3. Second subdiagonal
    if rho_2 > 0.0 and n >= 3:
        ab[2, :-2] = rho_2

    cb = la.cholesky_banded(ab, lower=True)
    return cb, True


class MultiOrderTrendFilterEstimator(BaseDeltaEstimator):
    """Offline Batch Multi-Order Trend Filtering MAP Estimator under Physical Bounds."""

    def __init__(
        self,
        lambda_0: float = 0.05,
        lambda_1: float = 0.50,
        lambda_2: float = 1.00,
        target_channel: str = "L_T1",
        bounds: tuple[float, float] | None = None,
        channel_manifest_path: str | Path | None = None,
        surrogate_estimator: BaseDeltaEstimator | Any | None = None,
        residual_std: float | None = None,
        sigma_epsilon: float = 1.0,
        solver: SolverType = "admm",
        max_iter: int = 1000,
        tol: float = 1e-6,
        rho_0: float = 2.0,
        rho_1: float = 2.0,
        rho_2: float = 2.0,
        **kwargs: Any,
    ) -> None:
        # Support aliases
        if "lambda_l1" in kwargs and kwargs["lambda_l1"] is not None:
            lambda_0 = float(kwargs["lambda_l1"])
        if "lambda_tv" in kwargs and kwargs["lambda_tv"] is not None:
            lambda_1 = float(kwargs["lambda_tv"])
        if "lambda_tf" in kwargs and kwargs["lambda_tf"] is not None:
            lambda_2 = float(kwargs["lambda_tf"])

        if lambda_0 < 0.0:
            raise ValueError("lambda_0 must be non-negative")
        if lambda_1 < 0.0:
            raise ValueError("lambda_1 must be non-negative")
        if lambda_2 < 0.0:
            raise ValueError("lambda_2 must be non-negative")
        if solver not in ("admm", "scipy"):
            raise ValueError("solver must be either 'admm' or 'scipy'")
        if sigma_epsilon <= 0.0:
            raise ValueError("sigma_epsilon must be positive")

        self.lambda_0 = float(lambda_0)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.target_channel = target_channel
        self.surrogate_estimator = surrogate_estimator
        self._residual_std = float(residual_std) if residual_std is not None else None
        self.sigma_epsilon = float(sigma_epsilon)
        self.solver: SolverType = solver
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rho_0 = float(rho_0)
        self.rho_1 = float(rho_1)
        self.rho_2 = float(rho_2)

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
    ) -> MultiOrderTrendFilterEstimator:
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
        """Reconstruct clean signal and attack trajectory via MAP optimization."""
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
                provenance={"estimator": self.__class__.__name__},
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
        if surr_unc == 0.0:
            surr_unc = self.sigma_epsilon

        # 3. Determine box bounds on delta: y - x_max <= delta <= y - x_min
        x_min, x_max = self.bounds
        lower = np.full(n, -np.inf, dtype=np.float64)
        upper = np.full(n, np.inf, dtype=np.float64)

        if np.isfinite(x_max):
            lower = y - x_max
        if np.isfinite(x_min):
            upper = y - x_min

        if attack_interval is not None:
            start, stop = attack_interval
            if not (0 <= start <= stop <= n):
                raise ValueError(f"Invalid attack_interval {attack_interval} for length {n}")
            lower[:start] = 0.0
            upper[:start] = 0.0
            lower[stop:] = 0.0
            upper[stop:] = 0.0

        lower = np.minimum(lower, upper)
        residual = y - x_surr

        # 4. Solve MAP optimization
        if self.solver == "admm":
            delta_hat, iters, loss = self._solve_admm(residual, lower, upper)
        else:
            delta_hat, iters, loss = self._solve_scipy(residual, lower, upper)

        # Strict safety projection
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
            "lambda_0": self.lambda_0,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "sigma_epsilon": self.sigma_epsilon,
            "bounds": self.bounds,
            "iterations": iters,
            "objective_loss": exact_loss,
            "residual_std": surr_unc,
        }

        unc_arr = np.full(n, surr_unc, dtype=np.float64)
        var_arr = np.full(n, surr_unc**2, dtype=np.float64)

        return ReconstructionResult(
            x_hat=x_hat,
            delta_hat=delta_hat,
            pointwise_uncertainty=unc_arr,
            pointwise_variance=var_arr,
            objective_loss=exact_loss,
            ambiguity_flag=False,
            provenance=prov,
        )

    def compute_objective(self, residual: np.ndarray, delta: np.ndarray) -> float:
        """Compute the deterministic scalar MAP objective."""
        w_eps = 1.0 / (self.sigma_epsilon ** 2)
        data_fidelity = 0.5 * w_eps * np.sum((residual - delta) ** 2)
        l0_penalty = self.lambda_0 * np.sum(np.abs(delta))
        l1_penalty = self.lambda_1 * np.sum(np.abs(np.diff(delta))) if len(delta) > 1 and self.lambda_1 > 0 else 0.0
        l2_penalty = self.lambda_2 * np.sum(np.abs(np.diff(delta, n=2))) if len(delta) > 2 and self.lambda_2 > 0 else 0.0
        return float(data_fidelity + l0_penalty + l1_penalty + l2_penalty)

    def _solve_admm(
        self,
        residual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        warm_delta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Solve bounded Multi-Order Trend Filtering MAP via ADMM banded Cholesky solves."""
        n = len(residual)
        if n == 1:
            best_d = float(np.clip(residual[0], lower[0], upper[0]))
            return np.array([best_d]), 1, self.compute_objective(residual, np.array([best_d]))

        w_eps = 1.0 / (self.sigma_epsilon ** 2)
        rho_0 = self.rho_0
        rho_1 = self.rho_1 if self.lambda_1 > 0 else 0.0
        rho_2 = self.rho_2 if (self.lambda_2 > 0 and n >= 3) else 0.0

        cho_factors = _build_pentadiagonal_cholesky(n, w_eps, rho_0, rho_1, rho_2)

        if warm_delta is not None and len(warm_delta) == n:
            delta = np.clip(warm_delta, lower, upper)
        else:
            delta = np.clip(residual, lower, upper)

        z0 = delta.copy()
        z1 = np.diff(delta) if rho_1 > 0 else np.zeros(n - 1, dtype=np.float64)
        z2 = np.diff(delta, n=2) if rho_2 > 0 else np.zeros(n - 2, dtype=np.float64)

        u0 = np.zeros(n, dtype=np.float64)
        u1 = np.zeros(n - 1, dtype=np.float64)
        u2 = np.zeros(n - 2, dtype=np.float64)

        iter_count = 0
        for it in range(self.max_iter):
            iter_count = it + 1

            # 1. Primal delta update
            rhs = w_eps * residual + rho_0 * (z0 - u0)
            if rho_1 > 0:
                v1 = z1 - u1
                dt_v1 = np.zeros(n, dtype=np.float64)
                dt_v1[:-1] -= v1
                dt_v1[1:] += v1
                rhs += rho_1 * dt_v1
            if rho_2 > 0:
                v2 = z2 - u2
                dt_v2 = np.zeros(n, dtype=np.float64)
                dt_v2[:-2] += v2
                dt_v2[1:-1] -= 2.0 * v2
                dt_v2[2:] += v2
                rhs += rho_2 * dt_v2

            delta_new = la.cho_solve_banded(cho_factors, rhs)

            # 2. Auxiliary updates
            z0 = np.clip(_soft_threshold(delta_new + u0, self.lambda_0 / rho_0), lower, upper)

            d1_delta = np.diff(delta_new) if (self.lambda_1 > 0 or rho_1 > 0) else np.zeros(n - 1, dtype=np.float64)
            if rho_1 > 0:
                z1 = _soft_threshold(d1_delta + u1, self.lambda_1 / rho_1)

            d2_delta = np.diff(delta_new, n=2) if (self.lambda_2 > 0 or rho_2 > 0) else np.zeros(n - 2, dtype=np.float64)
            if rho_2 > 0:
                z2 = _soft_threshold(d2_delta + u2, self.lambda_2 / rho_2)

            # 3. Dual updates
            u0 += delta_new - z0
            if rho_1 > 0:
                u1 += d1_delta - z1
            if rho_2 > 0:
                u2 += d2_delta - z2

            # 4. Convergence check
            primal_res = np.max(np.abs(delta_new - z0))
            if rho_1 > 0:
                primal_res = max(primal_res, float(np.max(np.abs(d1_delta - z1))))
            if rho_2 > 0:
                primal_res = max(primal_res, float(np.max(np.abs(d2_delta - z2))))

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
        """Solve bounded Multi-Order Trend Filtering MAP via Scipy optimization."""
        n = len(residual)
        if n == 1:
            best_d = float(np.clip(residual[0], lower[0], upper[0]))
            return np.array([best_d]), 1, self.compute_objective(residual, np.array([best_d]))

        w_eps = 1.0 / (self.sigma_epsilon ** 2)
        l0, l1, l2 = self.lambda_0, self.lambda_1, self.lambda_2

        D1 = np.diff(np.eye(n), axis=0) if l1 > 0 else None
        D2 = np.diff(np.eye(n), n=2, axis=0) if (l2 > 0 and n >= 3) else None

        n_dp = n if l0 > 0 else 0
        n_p1 = (n - 1) if l1 > 0 else 0
        n_p2 = (n - 2) if (l2 > 0 and n >= 3) else 0

        n_vars = n + 2 * n_dp + 2 * n_p1 + 2 * n_p2

        def qp_obj(z: np.ndarray) -> float:
            d = z[:n]
            idx = n
            loss = 0.5 * w_eps * np.sum((residual - d) ** 2)
            if l0 > 0:
                dp, dm = z[idx : idx + n], z[idx + n : idx + 2 * n]
                loss += l0 * np.sum(dp + dm)
                idx += 2 * n
            if l1 > 0:
                p1, q1 = z[idx : idx + n - 1], z[idx + n - 1 : idx + 2 * n - 2]
                loss += l1 * np.sum(p1 + q1)
                idx += 2 * (n - 1)
            if l2 > 0 and n >= 3:
                p2, q2 = z[idx : idx + n - 2], z[idx + n - 2 : idx + 2 * n - 4]
                loss += l2 * np.sum(p2 + q2)
            return float(loss)

        bnds = [(lower[i], upper[i]) for i in range(n)] + [(0.0, None)] * (n_vars - n)
        cons = []
        idx = n
        if l0 > 0:
            def c0(z: np.ndarray) -> np.ndarray:
                return z[:n] - z[n : 2 * n] + z[2 * n : 3 * n]
            cons.append({"type": "eq", "fun": c0})
            idx += 2 * n
        if l1 > 0 and D1 is not None:
            idx1 = idx
            def c1(z: np.ndarray) -> np.ndarray:
                return D1 @ z[:n] - z[idx1 : idx1 + n - 1] + z[idx1 + n - 1 : idx1 + 2 * n - 2]
            cons.append({"type": "eq", "fun": c1})
            idx += 2 * (n - 1)
        if l2 > 0 and n >= 3 and D2 is not None:
            idx2 = idx
            def c2(z: np.ndarray) -> np.ndarray:
                return D2 @ z[:n] - z[idx2 : idx2 + n - 2] + z[idx2 + n - 2 : idx2 + 2 * n - 4]
            cons.append({"type": "eq", "fun": c2})

        z0 = np.zeros(n_vars, dtype=np.float64)
        z0[:n] = np.clip(residual, lower, upper)
        idx = n
        if l0 > 0:
            z0[idx : idx + n] = np.maximum(z0[:n], 0.0)
            z0[idx + n : idx + 2 * n] = np.maximum(-z0[:n], 0.0)
            idx += 2 * n
        if l1 > 0 and D1 is not None:
            d1 = D1 @ z0[:n]
            z0[idx : idx + n - 1] = np.maximum(d1, 0.0)
            z0[idx + n - 1 : idx + 2 * n - 2] = np.maximum(-d1, 0.0)
            idx += 2 * (n - 1)
        if l2 > 0 and n >= 3 and D2 is not None:
            d2 = D2 @ z0[:n]
            z0[idx : idx + n - 2] = np.maximum(d2, 0.0)
            z0[idx + n - 2 : idx + 2 * n - 4] = np.maximum(-d2, 0.0)

        res = minimize(
            qp_obj,
            z0,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        delta_hat = res.x[:n]
        return delta_hat, int(res.nit), self.compute_objective(residual, delta_hat)


class CausalMultiOrderTrendFilterEstimator(BaseDeltaEstimator):
    """Past-Only Strictly Causal Rolling Multi-Order Trend Filter for Additive Attack Recovery."""

    def __init__(
        self,
        window_size: int = 48,
        lambda_0: float = 0.05,
        lambda_1: float = 0.50,
        lambda_2: float = 1.00,
        target_channel: str = "L_T1",
        bounds: tuple[float, float] | None = None,
        channel_manifest_path: str | Path | None = None,
        surrogate_estimator: BaseDeltaEstimator | Any | None = None,
        residual_std: float | None = None,
        sigma_epsilon: float = 1.0,
        solver: SolverType = "admm",
        max_iter: int = 300,
        tol: float = 1e-5,
        rho_0: float = 2.0,
        rho_1: float = 2.0,
        rho_2: float = 2.0,
        **kwargs: Any,
    ) -> None:
        # Support aliases
        if "lambda_l1" in kwargs and kwargs["lambda_l1"] is not None:
            lambda_0 = float(kwargs["lambda_l1"])
        if "lambda_tv" in kwargs and kwargs["lambda_tv"] is not None:
            lambda_1 = float(kwargs["lambda_tv"])
        if "lambda_tf" in kwargs and kwargs["lambda_tf"] is not None:
            lambda_2 = float(kwargs["lambda_tf"])

        if window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if lambda_0 < 0.0:
            raise ValueError("lambda_0 must be non-negative")
        if lambda_1 < 0.0:
            raise ValueError("lambda_1 must be non-negative")
        if lambda_2 < 0.0:
            raise ValueError("lambda_2 must be non-negative")
        if solver not in ("admm", "scipy"):
            raise ValueError("solver must be either 'admm' or 'scipy'")
        if sigma_epsilon <= 0.0:
            raise ValueError("sigma_epsilon must be positive")

        self.window_size = int(window_size)
        self.lambda_0 = float(lambda_0)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.target_channel = target_channel
        self.surrogate_estimator = surrogate_estimator
        self._residual_std = float(residual_std) if residual_std is not None else None
        self.sigma_epsilon = float(sigma_epsilon)
        self.solver: SolverType = solver
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rho_0 = float(rho_0)
        self.rho_1 = float(rho_1)
        self.rho_2 = float(rho_2)

        # Cholesky factorization cache for ADMM pentadiagonal system: {window_len: (cb, lower)}
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
    ) -> CausalMultiOrderTrendFilterEstimator:
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

    def _get_cholesky_factors(self, n: int) -> tuple[np.ndarray, bool]:
        """Obtain or compute cached Cholesky factors for pentadiagonal system."""
        if n in self._cho_cache:
            return self._cho_cache[n]

        w_eps = 1.0 / (self.sigma_epsilon ** 2)
        rho_0 = self.rho_0
        rho_1 = self.rho_1 if self.lambda_1 > 0 else 0.0
        rho_2 = self.rho_2 if (self.lambda_2 > 0 and n >= 3) else 0.0

        factors = _build_pentadiagonal_cholesky(n, w_eps, rho_0, rho_1, rho_2)
        self._cho_cache[n] = factors
        return factors

    def compute_objective(self, residual: np.ndarray, delta: np.ndarray) -> float:
        """Compute the deterministic scalar MAP objective."""
        w_eps = 1.0 / (self.sigma_epsilon ** 2)
        data_fidelity = 0.5 * w_eps * np.sum((residual - delta) ** 2)
        l0_penalty = self.lambda_0 * np.sum(np.abs(delta))
        l1_penalty = self.lambda_1 * np.sum(np.abs(np.diff(delta))) if len(delta) > 1 and self.lambda_1 > 0 else 0.0
        l2_penalty = self.lambda_2 * np.sum(np.abs(np.diff(delta, n=2))) if len(delta) > 2 and self.lambda_2 > 0 else 0.0
        return float(data_fidelity + l0_penalty + l1_penalty + l2_penalty)

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
        if surr_unc == 0.0:
            surr_unc = self.sigma_epsilon

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

            # Solve MAP on rolling window
            if w_len == 1:
                sol = np.clip(r_w, lower_w, upper_w)
                step_loss = self.compute_objective(r_w, sol)
            elif self.solver == "admm":
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
            "lambda_0": self.lambda_0,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
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

    def _solve_admm(
        self,
        residual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        warm_delta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Solve bounded Multi-Order Trend Filtering MAP via ADMM with cached Cholesky solves."""
        n = len(residual)
        if n == 1:
            best_d = float(np.clip(residual[0], lower[0], upper[0]))
            return np.array([best_d]), 1, self.compute_objective(residual, np.array([best_d]))

        w_eps = 1.0 / (self.sigma_epsilon ** 2)
        rho_0 = self.rho_0
        rho_1 = self.rho_1 if self.lambda_1 > 0 else 0.0
        rho_2 = self.rho_2 if (self.lambda_2 > 0 and n >= 3) else 0.0

        cho_factors = self._get_cholesky_factors(n)

        if warm_delta is not None and len(warm_delta) == n:
            delta = np.clip(warm_delta, lower, upper)
        else:
            delta = np.clip(residual, lower, upper)

        z0 = delta.copy()
        z1 = np.diff(delta) if rho_1 > 0 else np.zeros(n - 1, dtype=np.float64)
        z2 = np.diff(delta, n=2) if rho_2 > 0 else np.zeros(n - 2, dtype=np.float64)

        u0 = np.zeros(n, dtype=np.float64)
        u1 = np.zeros(n - 1, dtype=np.float64)
        u2 = np.zeros(n - 2, dtype=np.float64)

        iter_count = 0
        for it in range(self.max_iter):
            iter_count = it + 1

            # 1. Primal delta update
            rhs = w_eps * residual + rho_0 * (z0 - u0)
            if rho_1 > 0:
                v1 = z1 - u1
                dt_v1 = np.zeros(n, dtype=np.float64)
                dt_v1[:-1] -= v1
                dt_v1[1:] += v1
                rhs += rho_1 * dt_v1
            if rho_2 > 0:
                v2 = z2 - u2
                dt_v2 = np.zeros(n, dtype=np.float64)
                dt_v2[:-2] += v2
                dt_v2[1:-1] -= 2.0 * v2
                dt_v2[2:] += v2
                rhs += rho_2 * dt_v2

            delta_new = la.cho_solve_banded(cho_factors, rhs)

            # 2. Auxiliary updates
            z0 = np.clip(_soft_threshold(delta_new + u0, self.lambda_0 / rho_0), lower, upper)

            d1_delta = np.diff(delta_new) if (self.lambda_1 > 0 or rho_1 > 0) else np.zeros(n - 1, dtype=np.float64)
            if rho_1 > 0:
                z1 = _soft_threshold(d1_delta + u1, self.lambda_1 / rho_1)

            d2_delta = np.diff(delta_new, n=2) if (self.lambda_2 > 0 or rho_2 > 0) else np.zeros(n - 2, dtype=np.float64)
            if rho_2 > 0:
                z2 = _soft_threshold(d2_delta + u2, self.lambda_2 / rho_2)

            # 3. Dual updates
            u0 += delta_new - z0
            if rho_1 > 0:
                u1 += d1_delta - z1
            if rho_2 > 0:
                u2 += d2_delta - z2

            # 4. Convergence check
            primal_res = np.max(np.abs(delta_new - z0))
            if rho_1 > 0:
                primal_res = max(primal_res, float(np.max(np.abs(d1_delta - z1))))
            if rho_2 > 0:
                primal_res = max(primal_res, float(np.max(np.abs(d2_delta - z2))))

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
        """Solve bounded Multi-Order Trend Filtering MAP via Scipy optimization."""
        n = len(residual)
        if n == 1:
            best_d = float(np.clip(residual[0], lower[0], upper[0]))
            return np.array([best_d]), 1, self.compute_objective(residual, np.array([best_d]))

        w_eps = 1.0 / (self.sigma_epsilon ** 2)
        l0, l1, l2 = self.lambda_0, self.lambda_1, self.lambda_2

        D1 = np.diff(np.eye(n), axis=0) if l1 > 0 else None
        D2 = np.diff(np.eye(n), n=2, axis=0) if (l2 > 0 and n >= 3) else None

        n_dp = n if l0 > 0 else 0
        n_p1 = (n - 1) if l1 > 0 else 0
        n_p2 = (n - 2) if (l2 > 0 and n >= 3) else 0

        n_vars = n + 2 * n_dp + 2 * n_p1 + 2 * n_p2

        def qp_obj(z: np.ndarray) -> float:
            d = z[:n]
            idx = n
            loss = 0.5 * w_eps * np.sum((residual - d) ** 2)
            if l0 > 0:
                dp, dm = z[idx : idx + n], z[idx + n : idx + 2 * n]
                loss += l0 * np.sum(dp + dm)
                idx += 2 * n
            if l1 > 0:
                p1, q1 = z[idx : idx + n - 1], z[idx + n - 1 : idx + 2 * n - 2]
                loss += l1 * np.sum(p1 + q1)
                idx += 2 * (n - 1)
            if l2 > 0 and n >= 3:
                p2, q2 = z[idx : idx + n - 2], z[idx + n - 2 : idx + 2 * n - 4]
                loss += l2 * np.sum(p2 + q2)
            return float(loss)

        bnds = [(lower[i], upper[i]) for i in range(n)] + [(0.0, None)] * (n_vars - n)
        cons = []
        idx = n
        if l0 > 0:
            def c0(z: np.ndarray) -> np.ndarray:
                return z[:n] - z[n : 2 * n] + z[2 * n : 3 * n]
            cons.append({"type": "eq", "fun": c0})
            idx += 2 * n
        if l1 > 0 and D1 is not None:
            idx1 = idx
            def c1(z: np.ndarray) -> np.ndarray:
                return D1 @ z[:n] - z[idx1 : idx1 + n - 1] + z[idx1 + n - 1 : idx1 + 2 * n - 2]
            cons.append({"type": "eq", "fun": c1})
            idx += 2 * (n - 1)
        if l2 > 0 and n >= 3 and D2 is not None:
            idx2 = idx
            def c2(z: np.ndarray) -> np.ndarray:
                return D2 @ z[:n] - z[idx2 : idx2 + n - 2] + z[idx2 + n - 2 : idx2 + 2 * n - 4]
            cons.append({"type": "eq", "fun": c2})

        z0 = np.zeros(n_vars, dtype=np.float64)
        z0[:n] = np.clip(residual, lower, upper)
        idx = n
        if l0 > 0:
            z0[idx : idx + n] = np.maximum(z0[:n], 0.0)
            z0[idx + n : idx + 2 * n] = np.maximum(-z0[:n], 0.0)
            idx += 2 * n
        if l1 > 0 and D1 is not None:
            d1 = D1 @ z0[:n]
            z0[idx : idx + n - 1] = np.maximum(d1, 0.0)
            z0[idx + n - 1 : idx + 2 * n - 2] = np.maximum(-d1, 0.0)
            idx += 2 * (n - 1)
        if l2 > 0 and n >= 3 and D2 is not None:
            d2 = D2 @ z0[:n]
            z0[idx : idx + n - 2] = np.maximum(d2, 0.0)
            z0[idx + n - 2 : idx + 2 * n - 4] = np.maximum(-d2, 0.0)

        res = minimize(
            qp_obj,
            z0,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        delta_hat = res.x[:n]
        return delta_hat, int(res.nit), self.compute_objective(residual, delta_hat)
