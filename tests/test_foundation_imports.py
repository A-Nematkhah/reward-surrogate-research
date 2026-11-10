"""Minimal import smoke tests for Step 1 foundation."""

from __future__ import annotations


def test_import_rsr_package() -> None:
    import rsr

    assert hasattr(rsr, "__version__")
    assert isinstance(rsr.__version__, str)


def test_import_utils_package() -> None:
    from rsr import utils
    from rsr.utils import (
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
    import rsr.dataset
    import rsr.environment
    import rsr.evaluation
    import rsr.features
    import rsr.rewards
    import rsr.surrogate
    import rsr.training
    import rsr.utils

    assert rsr.environment.__doc__
    assert rsr.rewards.__doc__
    assert rsr.training.__doc__
    assert rsr.evaluation.__doc__
    assert rsr.dataset.__doc__
    assert rsr.features.__doc__
    assert rsr.surrogate.__doc__
    assert rsr.utils.__doc__
