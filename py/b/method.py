from enum import Enum


class Method(Enum):
    CD = 1, False
    MCLV = 2, True
    PCD = 4, False

    @staticmethod
    def requires_warmup(method):
        return method.value[1]


if __name__ == '__main__':
    for method in Method:
        print(method.value)
