"""ManiSkill helpers shared by the JAX and torch trainers."""

# Tasks whose scene builders edit render components (PushT un-textures the table), which
# do not exist when `render_backend: none` disables the render device.
_TASKS_NEEDING_RENDER = ("PushT-v1",)


def maniskill_env_kwargs(env_name: str, env_kwargs) -> dict:
    """`env_kwargs` for `gym.make`, with rendering kept on for tasks that need it to build."""
    kwargs = dict(env_kwargs or {})
    if env_name in _TASKS_NEEDING_RENDER and kwargs.get("render_backend") == "none":
        kwargs["render_backend"] = "gpu"
    return kwargs
