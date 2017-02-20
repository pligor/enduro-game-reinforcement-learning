import numpy as np
from enduro_data_types import tuple_dt

def generateStates(categories, getInitialTuple, tuples_dt):
    shift = lambda arr: (arr[0], arr[1:])

    tupleList = []

    def func(categs, tuple):
        assert len(categs) > 0

        curCateg, restOfCategs = shift(categs)

        # print len(curCateg.value)
        for v in curCateg.value:
            # print v
            curTuple = tuple.copy()

            curTuple[curCateg.name] = v

            if len(restOfCategs) == 0:
                tupleList.append(curTuple)
            else:
                func(restOfCategs, curTuple)

    func(categories, getInitialTuple())

    return np.array([(i, tuple) for i, tuple in enumerate(tupleList)], dtype=tuples_dt)
