# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict JSON loading shared by optimization corpus artifacts."""

import json
from pathlib import Path
from typing import Any


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def load_json(path: Path) -> Any:
    """Load strict JSON, rejecting duplicate keys and non-finite numbers."""

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's boolean/integer coercion."""

    return json.dumps(
        left,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) == json.dumps(
        right,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
