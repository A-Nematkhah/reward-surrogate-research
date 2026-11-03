"""Reward Surrogate Research package."""

__version__ = "0.1.0"


def _block_tensorflow_probe() -> None:
    """Stop TensorBoard from importing a TensorFlow that is broken under NumPy 2.

    ``torch.utils.tensorboard`` (imported by stable-baselines3) makes
    ``tensorboard.compat`` probe for a ``notf`` marker module and, when absent,
    import TensorFlow. An old TensorFlow built against NumPy 1.x raises a
    ``TypeError`` / ``_ARRAY_API`` error there, which kills ``import
    stable_baselines3``. This project never uses TensorFlow
    (``tensorboard_log=None``), so we register the marker and TensorBoard
    falls back to its own lightweight stub instead.
    """
    import os
    import sys
    import types

    if os.environ.get("RSR_ALLOW_TENSORFLOW") == "1":
        return
    sys.modules.setdefault(
        "tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf")
    )


_block_tensorflow_probe()
