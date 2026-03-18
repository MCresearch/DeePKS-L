"""Adapter implementations wiring contracts to concrete backends."""

__all__ = ["CorrNetModelBackend", "PySCFPhysicsBackend"]


def __getattr__(name):
	if name == "CorrNetModelBackend":
		from .model_backend import CorrNetModelBackend

		return CorrNetModelBackend
	if name == "PySCFPhysicsBackend":
		from .physics_backend import PySCFPhysicsBackend

		return PySCFPhysicsBackend
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
