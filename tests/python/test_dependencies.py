from __future__ import annotations

import importlib.metadata as metadata

from packaging.requirements import Requirement


def test_default_python_dependencies_stay_polars_only() -> None:
    requirements = [
        Requirement(requirement)
        for requirement in (metadata.requires("ferric-alpha") or [])
    ]
    unconditional = [
        requirement for requirement in requirements if requirement.marker is None
    ]
    assert [str(requirement) for requirement in unconditional] == ["polars<1.43,>=1.42"]

    extras = {
        requirement.marker.evaluate({"extra": extra}): str(requirement)
        for extra in ["plot", "notebook"]
        for requirement in requirements
        if requirement.marker is not None
    }
    assert extras[True]


def test_optional_dependency_metadata_is_exact() -> None:
    requirements = [
        Requirement(requirement)
        for requirement in (metadata.requires("ferric-alpha") or [])
    ]

    def active(extra: str) -> list[str]:
        return sorted(
            str(requirement)
            for requirement in requirements
            if requirement.marker is not None
            and requirement.marker.evaluate({"extra": extra})
        )

    assert active("plot") == ['matplotlib<3.11,>=3.8; extra == "plot"']
    assert active("notebook") == ['ipython<10,>=8; extra == "notebook"']
