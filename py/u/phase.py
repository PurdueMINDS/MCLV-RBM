from enum import Enum, auto


class Phase(Enum):
    DATA = 1
    GRADIENT = 2
    RUN_TOURS = 3