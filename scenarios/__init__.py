"""Scenario registry — imports and registers all scenario classes."""
from .base import ScenarioBase, ScenarioRegistry, GroundTruth
from .easy_oom_crash import EasyOOMCrash
from .medium_cascade import MediumCascade
from .medium_disk_full import MediumDiskFull
from .hard_intermittent import HardIntermittent
from .hard_network_partition import HardNetworkPartition

# Register all scenarios
ScenarioRegistry.register("easy_oom_crash", EasyOOMCrash)
ScenarioRegistry.register("medium_cascade", MediumCascade)
ScenarioRegistry.register("medium_disk_full", MediumDiskFull)
ScenarioRegistry.register("hard_intermittent", HardIntermittent)
ScenarioRegistry.register("hard_network_partition", HardNetworkPartition)

__all__ = [
    "ScenarioBase",
    "ScenarioRegistry",
    "GroundTruth",
    "EasyOOMCrash",
    "MediumCascade",
    "MediumDiskFull",
    "HardIntermittent",
    "HardNetworkPartition",
]
