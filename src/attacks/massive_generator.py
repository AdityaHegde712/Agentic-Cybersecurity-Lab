"""High-Throughput Attack Scenario Generator for Supervisory Sensor Telemetry.

Synthesizes attack trajectories across canonical additive profiles, complex physical
waveforms, and stochastic non-deterministic drift processes. Supports parameterized
sweeps, multi-channel injection, physical envelope checking, and columnar Parquet export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttackScenario:
    """Immutable specification and realization of an additive sensor attack scenario."""

    scenario_id: str
    family: str
    target_channels: list[str]
    onset_step: int
    duration_steps: int
    amplitude: float
    parameters: dict[str, Any]
    delta: np.ndarray
    support_mask: np.ndarray

    def __post_init__(self) -> None:
        """Validate array dimensions and consistency."""
        if self.delta.shape != self.support_mask.shape:
            raise ValueError(
                f"delta shape {self.delta.shape} must match support_mask shape {self.support_mask.shape}"
            )


class AttackBatchGenerator:
    """Vectorized attack scenario synthesizer and batch generator."""

    SUPPORTED_FAMILIES: set[str] = {
        "step",
        "ramp",
        "drift",
        "pulse",
        "periodic",
        "sinusoid",
        "trapezoid",
        "sigmoidal",
        "sigmoid",
        "chirp",
        "brownian",
        "random_walk",
        "ou",
        "ornstein_uhlenbeck",
    }

    CANONICAL_FAMILY_MAP: dict[str, str] = {
        "sinusoid": "periodic",
        "sigmoid": "sigmoidal",
        "random_walk": "brownian",
        "ornstein_uhlenbeck": "ou",
    }

    def __init__(
        self,
        base_telemetry: np.ndarray,
        channel_names: list[str],
        seed: int = 42,
    ) -> None:
        """Initialize the attack batch generator.

        Args:
            base_telemetry: Array of shape [N, d] representing baseline uncorrupted telemetry.
            channel_names: List of d unique sensor channel names.
            seed: Master random seed for deterministic reproducibility.
        """
        if base_telemetry.ndim != 2:
            raise ValueError(
                "base_telemetry must be a 2D array [n_timesteps, n_channels]"
            )
        if base_telemetry.shape[0] == 0 or base_telemetry.shape[1] == 0:
            raise ValueError("base_telemetry must have non-zero timesteps and channels")
        if not np.all(np.isfinite(base_telemetry)):
            raise ValueError("base_telemetry must contain only finite numerical values")
        if len(channel_names) != base_telemetry.shape[1]:
            raise ValueError(
                f"channel_names length ({len(channel_names)}) must match telemetry columns ({base_telemetry.shape[1]})"
            )
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("channel_names must not contain duplicates")

        self.base_telemetry = np.asarray(base_telemetry, dtype=np.float64)
        self.n_steps, self.n_channels = self.base_telemetry.shape
        self.channel_names = list(channel_names)
        self._channel_to_idx = {name: idx for idx, name in enumerate(channel_names)}
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _normalize_family(self, family: str) -> str:
        """Map aliases to canonical family names and validate."""
        clean = family.strip().lower()
        if clean not in self.SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported family '{family}'. Supported families: {sorted(self.SUPPORTED_FAMILIES)}"
            )
        return self.CANONICAL_FAMILY_MAP.get(clean, clean)

    def _generate_waveform(
        self,
        family: str,
        duration: int,
        amplitude: float,
        parameters: dict[str, Any],
    ) -> np.ndarray:
        """Synthesize a 1D perturbation waveform of length duration."""
        if duration <= 0:
            raise ValueError(f"duration_steps must be positive, got {duration}")

        canonical_fam = self._normalize_family(family)

        if canonical_fam == "step":
            return np.full(duration, amplitude, dtype=np.float64)

        if canonical_fam == "ramp":
            if duration == 1:
                return np.full(1, amplitude, dtype=np.float64)
            return amplitude * (np.arange(duration, dtype=np.float64) / (duration - 1))

        if canonical_fam == "drift":
            rate = parameters.get("rate")
            if rate is not None:
                return np.arange(duration, dtype=np.float64) * float(rate)
            if duration == 1:
                return np.full(1, amplitude, dtype=np.float64)
            return amplitude * (np.arange(duration, dtype=np.float64) / (duration - 1))

        if canonical_fam == "pulse":
            pulse_width = int(parameters.get("pulse_width", duration))
            pulse = np.zeros(duration, dtype=np.float64)
            pulse[: min(duration, max(0, pulse_width))] = amplitude
            return pulse

        if canonical_fam == "periodic":
            period = float(parameters.get("period", 24.0))
            phase = float(parameters.get("phase", 0.0))
            t = np.arange(duration, dtype=np.float64)
            return amplitude * np.sin(2.0 * np.pi * t / period + phase)

        if canonical_fam == "trapezoid":
            if duration == 1:
                return np.full(1, amplitude, dtype=np.float64)
            u1 = float(parameters.get("u1", 0.2))
            u2 = float(parameters.get("u2", 0.8))
            u = np.arange(duration, dtype=np.float64) / (duration - 1)
            psi = np.zeros(duration, dtype=np.float64)

            mask_rise = u < u1
            mask_hold = (u >= u1) & (u < u2)
            mask_decay = u >= u2

            if u1 > 0:
                psi[mask_rise] = u[mask_rise] / u1
            else:
                psi[mask_rise] = 1.0

            psi[mask_hold] = 1.0

            if u2 < 1.0:
                psi[mask_decay] = (1.0 - u[mask_decay]) / (1.0 - u2)
            else:
                psi[mask_decay] = 1.0

            return amplitude * psi

        if canonical_fam == "sigmoidal":
            if duration == 1:
                return np.full(1, amplitude, dtype=np.float64)
            k = float(parameters.get("k", 10.0 / max(1, duration - 1)))
            t0 = float(parameters.get("t0", (duration - 1) / 2.0))
            t = np.arange(duration, dtype=np.float64)

            arg = np.clip(k * (t - t0), -100.0, 100.0)
            g = 1.0 / (1.0 + np.exp(-arg))
            g0 = 1.0 / (1.0 + np.exp(np.clip(k * t0, -100.0, 100.0)))
            g1 = 1.0 / (1.0 + np.exp(-np.clip(k * (duration - 1 - t0), -100.0, 100.0)))

            if g1 > g0:
                norm_curve = (g - g0) / (g1 - g0)
            else:
                norm_curve = g
            return amplitude * norm_curve

        if canonical_fam == "chirp":
            f0 = float(parameters.get("f0", 0.01))
            f1 = float(parameters.get("f1", 0.1))
            t = np.arange(duration, dtype=np.float64)
            phase = 2.0 * np.pi * (f0 + (f1 - f0) / (2.0 * max(1, duration)) * t) * t
            return amplitude * np.sin(phase)

        if canonical_fam == "brownian":
            mu = float(parameters.get("mu", 0.0))
            sigma = parameters.get("sigma")
            if sigma is None:
                sigma = (
                    abs(amplitude) / np.sqrt(duration)
                    if duration > 0 and amplitude != 0.0
                    else 1.0
                )
            else:
                sigma = float(sigma)
            noise = self.rng.normal(loc=mu, scale=sigma, size=duration)
            walk = np.cumsum(noise)
            if parameters.get("normalize", False) and np.max(np.abs(walk)) > 0:
                walk = amplitude * (walk / np.max(np.abs(walk)))
            return walk

        if canonical_fam == "ou":
            theta = float(parameters.get("theta", 0.1))
            target = float(parameters.get("target", parameters.get("mu", amplitude)))
            sigma = float(
                parameters.get(
                    "sigma", 0.1 * abs(amplitude) if amplitude != 0.0 else 0.1
                )
            )
            dt = float(parameters.get("dt", 1.0))
            delta0 = float(parameters.get("delta0", 0.0))

            res = np.zeros(duration, dtype=np.float64)
            res[0] = delta0
            noise = self.rng.normal(0.0, 1.0, size=duration - 1)
            sqrt_dt = np.sqrt(dt)
            curr = delta0
            for i in range(duration - 1):
                curr = curr + theta * (target - curr) * dt + sigma * sqrt_dt * noise[i]
                res[i + 1] = curr
            return res

        raise ValueError(f"Unhandled canonical family: {canonical_fam}")

    def generate_scenario(
        self,
        scenario_id: str,
        family: str,
        target_channels: list[str] | str,
        onset_step: int,
        duration_steps: int,
        amplitude: float,
        parameters: dict[str, Any] | None = None,
    ) -> AttackScenario:
        """Synthesize a single attack scenario.

        Args:
            scenario_id: Unique string identifier.
            family: Attack morphology family.
            target_channels: Sensor channel name or list of sensor channel names.
            onset_step: Starting timestep of injection.
            duration_steps: Timesteps duration of active injection.
            amplitude: Perturbation amplitude scale.
            parameters: Additional family-specific parameters.

        Returns:
            AttackScenario instance containing delta and support_mask.
        """
        if not scenario_id:
            raise ValueError("scenario_id must not be empty")
        if onset_step < 0:
            raise ValueError(f"onset_step must be non-negative, got {onset_step}")
        if duration_steps <= 0:
            raise ValueError(f"duration_steps must be positive, got {duration_steps}")
        if onset_step + duration_steps > self.n_steps:
            raise ValueError(
                f"Attack interval [{onset_step}, {onset_step + duration_steps}) exceeds "
                f"telemetry length {self.n_steps}"
            )

        if isinstance(target_channels, str):
            channels = [target_channels]
        else:
            channels = list(target_channels)

        if not channels:
            raise ValueError("target_channels must not be empty")

        target_indices: list[int] = []
        for ch in channels:
            if ch not in self._channel_to_idx:
                raise ValueError(
                    f"Unknown channel '{ch}'. Available channels: {self.channel_names}"
                )
            target_indices.append(self._channel_to_idx[ch])

        canonical_fam = self._normalize_family(family)
        params = dict(parameters) if parameters is not None else {}

        waveform = self._generate_waveform(
            canonical_fam, duration_steps, amplitude, params
        )

        delta = np.zeros((self.n_steps, self.n_channels), dtype=np.float64)
        support_mask = np.zeros((self.n_steps, self.n_channels), dtype=bool)

        end_step = onset_step + duration_steps
        for idx in target_indices:
            delta[onset_step:end_step, idx] = waveform
            support_mask[onset_step:end_step, idx] = True

        return AttackScenario(
            scenario_id=scenario_id,
            family=canonical_fam,
            target_channels=channels,
            onset_step=onset_step,
            duration_steps=duration_steps,
            amplitude=amplitude,
            parameters=params,
            delta=delta,
            support_mask=support_mask,
        )

    def generate_batch(
        self,
        families: list[str] | None = None,
        channels: list[str] | None = None,
        onsets: list[int] | None = None,
        durations: list[int] | None = None,
        amplitudes: list[float] | None = None,
        n_scenarios: int | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> list[AttackScenario]:
        """Generate a batch of scenarios across parameterized sweeps or random sampling.

        Args:
            families: List of attack families.
            channels: List of sensor channel names.
            onsets: List of onset steps.
            durations: List of duration steps.
            amplitudes: List of amplitude values.
            n_scenarios: Total number of scenarios to generate.
            parameters: Shared parameters dictionary.

        Returns:
            List of generated AttackScenario instances.
        """
        fams = families if families is not None else ["step", "ramp", "periodic"]
        chs = channels if channels is not None else [self.channel_names[0]]
        ons = onsets if onsets is not None else [self.n_steps // 4, self.n_steps // 2]
        durs = durations if durations is not None else [20, 50]
        amps = amplitudes if amplitudes is not None else [1.0, 5.0]

        valid_max_dur = min(durs) if durs else 20
        max_onset = max(0, self.n_steps - valid_max_dur - 1)
        ons = [min(o, max_onset) for o in ons]

        scenarios: list[AttackScenario] = []

        if n_scenarios is not None:
            for i in range(n_scenarios):
                fam = fams[i % len(fams)]
                ch = chs[i % len(chs)]
                dur = durs[i % len(durs)]
                onset_limit = max(0, self.n_steps - dur)
                onset = ons[i % len(ons)]
                if onset + dur > self.n_steps:
                    onset = onset_limit // 2
                amp = amps[i % len(amps)]
                scen_id = f"batch_{i:06d}_{fam}"
                scen = self.generate_scenario(
                    scenario_id=scen_id,
                    family=fam,
                    target_channels=[ch],
                    onset_step=onset,
                    duration_steps=dur,
                    amplitude=amp,
                    parameters=parameters,
                )
                scenarios.append(scen)
        else:
            count = 0
            for fam in fams:
                for ch in chs:
                    for dur in durs:
                        for onset in ons:
                            if onset + dur > self.n_steps:
                                continue
                            for amp in amps:
                                scen_id = f"sweep_{count:06d}_{fam}"
                                scen = self.generate_scenario(
                                    scenario_id=scen_id,
                                    family=fam,
                                    target_channels=[ch],
                                    onset_step=onset,
                                    duration_steps=dur,
                                    amplitude=amp,
                                    parameters=parameters,
                                )
                                scenarios.append(scen)
                                count += 1

        return scenarios

    def generate_family(
        self,
        family: str,
        n_scenarios: int,
        param_distributions: dict[str, Any] | None = None,
    ) -> list[AttackScenario]:
        """Synthesize n_scenarios belonging strictly to a single family."""
        params_dist = param_distributions or {}
        scenarios: list[AttackScenario] = []
        for i in range(n_scenarios):
            ch = self.rng.choice(self.channel_names)
            dur = int(params_dist.get("duration", self.rng.integers(20, 100)))
            dur = min(dur, self.n_steps - 1)
            onset = int(
                params_dist.get("onset", self.rng.integers(0, self.n_steps - dur))
            )
            amp = float(params_dist.get("amplitude", self.rng.uniform(1.0, 10.0)))

            scen = self.generate_scenario(
                scenario_id=f"{family}_{i:05d}",
                family=family,
                target_channels=[ch],
                onset_step=onset,
                duration_steps=dur,
                amplitude=amp,
            )
            scenarios.append(scen)
        return scenarios

    def generate_massive_corpus(self, n_total: int = 5000) -> pd.DataFrame:
        """Generate a massive corpus split across all 10 morphology families."""
        canonical_families = [
            "step",
            "ramp",
            "drift",
            "pulse",
            "periodic",
            "trapezoid",
            "sigmoidal",
            "chirp",
            "brownian",
            "ou",
        ]
        n_per_family = max(1, n_total // len(canonical_families))
        all_scenarios: list[AttackScenario] = []
        for fam in canonical_families:
            all_scenarios.extend(self.generate_family(fam, n_per_family))

        return self.to_dataframe(all_scenarios[:n_total])

    def check_physical_envelope(
        self,
        scenario: AttackScenario,
        min_bounds: np.ndarray | dict[str, float] | None = None,
        max_bounds: np.ndarray | dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Validate scenario attacked telemetry against physical bounds.

        Checks: min_bounds <= (x_t + delta_t) <= max_bounds for all t and channels.

        Args:
            scenario: AttackScenario instance.
            min_bounds: Minimum allowable physical limits per channel.
            max_bounds: Maximum allowable physical limits per channel.

        Returns:
            Dictionary containing 'is_valid', 'violation_count', 'channels_violated',
            'min_violations', and 'max_violations'.
        """
        if min_bounds is None:
            min_arr = np.min(self.base_telemetry, axis=0)
        elif isinstance(min_bounds, dict):
            min_arr = np.array(
                [min_bounds.get(ch, -np.inf) for ch in self.channel_names],
                dtype=np.float64,
            )
        else:
            min_arr = np.asarray(min_bounds, dtype=np.float64)

        if max_bounds is None:
            max_arr = np.max(self.base_telemetry, axis=0)
        elif isinstance(max_bounds, dict):
            max_arr = np.array(
                [max_bounds.get(ch, np.inf) for ch in self.channel_names],
                dtype=np.float64,
            )
        else:
            max_arr = np.asarray(max_bounds, dtype=np.float64)

        observed = self.base_telemetry + scenario.delta

        min_viol = np.maximum(0.0, min_arr - observed)
        max_viol = np.maximum(0.0, observed - max_arr)

        viol_mask = (min_viol > 1e-9) | (max_viol > 1e-9)
        violation_count = int(np.sum(viol_mask))

        channels_violated: list[str] = []
        min_viol_dict: dict[str, float] = {}
        max_viol_dict: dict[str, float] = {}

        for ch_idx, ch_name in enumerate(self.channel_names):
            max_under = float(np.max(min_viol[:, ch_idx]))
            max_over = float(np.max(max_viol[:, ch_idx]))
            if max_under > 1e-9 or max_over > 1e-9:
                channels_violated.append(ch_name)
            if max_under > 1e-9:
                min_viol_dict[ch_name] = max_under
            if max_over > 1e-9:
                max_viol_dict[ch_name] = max_over

        return {
            "is_valid": violation_count == 0,
            "violation_count": violation_count,
            "channels_violated": channels_violated,
            "min_violations": min_viol_dict,
            "max_violations": max_viol_dict,
        }

    @staticmethod
    def to_dataframe(scenarios: list[AttackScenario]) -> pd.DataFrame:
        """Convert list of AttackScenario instances to tabular DataFrame ledger."""
        records: list[dict[str, Any]] = []
        for s in scenarios:
            delta_active = (
                s.delta[s.support_mask] if np.any(s.support_mask) else np.array([0.0])
            )
            records.append(
                {
                    "scenario_id": s.scenario_id,
                    "family": s.family,
                    "target_channels": ",".join(s.target_channels),
                    "target_channel_count": len(s.target_channels),
                    "onset_step": s.onset_step,
                    "duration_steps": s.duration_steps,
                    "amplitude": s.amplitude,
                    "parameters": json.dumps(s.parameters, default=str),
                    "l2_norm": float(np.linalg.norm(s.delta)),
                    "max_abs_delta": float(np.max(np.abs(delta_active))),
                    "support_fraction": float(np.mean(s.support_mask)),
                }
            )
        return pd.DataFrame.from_records(records)

    def save_parquet(
        self,
        scenarios: list[AttackScenario],
        output_path: Path | str,
    ) -> None:
        """Export scenario metadata ledger to Parquet format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe(scenarios)
        df.to_parquet(path, index=False)
