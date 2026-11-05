"""Import-time source patches for MJX internals; drop them once fixed upstream."""

import inspect
import logging

logger = logging.getLogger(__name__)

# XLA's algebraic simplifier miscompiles these two 3x3 contractions under a seed-vmap
# around the env-vmap (HLO verifier: "Failed after simplification"); as a broadcast
# multiply plus a reduction no `dot` is emitted. Bit-identical to the original.
_TOUCH_SENSOR_MATMULS = (
    ("pnt = (pos - xpos) @ xmat", "pnt = jp.sum((pos - xpos)[..., None] * xmat, -2)"),
    ("vec = conray @ xmat", "vec = jp.sum(conray[..., None] * xmat, -2)"),
)

_patched = False


def patch_touch_sensor_matmuls() -> bool:
    """Make `mjx.sensor_acc` survive a seed-vmap around an env-vmap.

    A no-op unless the model has touch sensors and `num_seeds > 1`. Idempotent;
    returns whether the patched version is in place.
    """
    global _patched
    if _patched:
        return True

    from mujoco.mjx._src import sensor

    source = inspect.getsource(sensor.sensor_acc)
    for original, replacement in _TOUCH_SENSOR_MATMULS:
        if original not in source:
            logger.warning(
                "MJX touch-sensor patch skipped: %r not found in sensor_acc "
                "(mujoco version drift?). num_seeds > 1 may fail to compile.",
                original,
            )
            return False
        source = source.replace(original, replacement)

    # `forward.py` resolves `sensor_acc` through the module, so rebinding it there is
    # enough; compiling against the module globals keeps upstream's names resolving.
    exec(compile(source, sensor.__file__, "exec"), sensor.__dict__)
    _patched = True
    return True
