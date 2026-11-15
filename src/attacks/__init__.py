"""Attack-generation utilities."""

from .additive import AdditiveAttackResult, AdditiveAttackSpec, inject_additive_attack
from .massive_generator import AttackBatchGenerator, AttackScenario

__all__ = [
    "AdditiveAttackResult",
    "AdditiveAttackSpec",
    "AttackBatchGenerator",
    "AttackScenario",
    "inject_additive_attack",
]
