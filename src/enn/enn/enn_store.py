from __future__ import annotations

from enum import Enum


class ENNStore(Enum):
    OLOGN = "OLOGN"
    O1 = "O1"


def store_name(store: ENNStore | str) -> str:
    return ENNStore(store).value if isinstance(store, str) else store.value
