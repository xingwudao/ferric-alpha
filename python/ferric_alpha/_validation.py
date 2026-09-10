from __future__ import annotations


def _u32_option(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 2**32 - 1
    ):
        raise ValueError(
            f"invalid option {name}: value must fit a non-negative 32-bit integer"
        )
    return value


def _u32_options(
    values: list[int] | tuple[int, ...] | None, name: str
) -> list[int] | None:
    if values is None:
        return None
    return [_u32_option(value, name) for value in values]
