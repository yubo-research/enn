from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ObsLists:
    x_obs: list
    y_obs: list
    y_tr: list
    yvar_obs: list
