"""Minimal import smoke tests for Step 1 foundation."""

from __future__ import annotations


def test_import_src_package() -> None:
    import src

    assert hasattr(src, "__version__")
    assert isinstance(src.__version__, str)


def test_import_utils_package() -> None:
    from src import utils
    from src.utils import (
        deep_update,
        get_logger,
        load_config,
        load_merged_configs,
        save_config,
        seed_everything,
        seed_worker,
        setup_logging,
    )

    assert utils is not None
    assert callable(load_config)
    assert callable(load_merged_configs)
    assert callable(save_config)
    assert callable(deep_update)
    assert callable(setup_logging)
    assert callable(get_logger)
    assert callable(seed_everything)
    assert callable(seed_worker)


def test_import_subpackages() -> None:
    """Package namespaces exist even before domain modules are implemented."""
    import src.dataset
    import src.environment
    import src.evaluation
    import src.features
    import src.rewards
    import src.surrogate
    import src.training
    import src.utils

    assert src.environment.__doc__
    assert src.rewards.__doc__
    assert src.training.__doc__
    assert src.evaluation.__doc__
    assert src.dataset.__doc__
    assert src.features.__doc__
    assert src.surrogate.__doc__
    assert src.utils.__doc__
