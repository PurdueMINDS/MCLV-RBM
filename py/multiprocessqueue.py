import os
import sys
from multiprocessing import Pool


def expensive_function(x):
    print(x)
    print(x, file=sys.stderr)
    return os.system(x)


if __name__ == '__main__':
    f = open("py/commands", 'r+')
    data = f.readlines()
    f.close()
    pool = Pool(processes=10)
    # evaluate "f(10)" asynchronously

    result = pool.map(expensive_function, data)
    print(result)
