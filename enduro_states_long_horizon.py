import os
import numpy as np
from road_category import RoadCategory
from extreme_position import ExtremePosition
from enduro_data_types import tuple_dt
from how_many_opponents import HowManyOpponents

def pop(arr):
    return arr[-1], arr[:-1]


def shift(arr):
    return arr[0], arr[1:]


class Category:
    def __init__(self, name, value):
        self.name = name
        self.value = value


road_categ = np.array(RoadCategory().getAll())
pos_categ = np.array(ExtremePosition().getAll())
one_car_ahead_approaching = np.array([False, True])
one_car_ahead_right_approaching = np.array([False, True])
one_car_ahead_left_approaching = np.array([False, True])
opponents_surpassing = np.array([False, True])
opponent_immediate_left = np.array([False, True])
opponent_immediate_right = np.array([False, True])
count_opp_right = np.array(HowManyOpponents().getAll())
count_opp_left = np.array(HowManyOpponents().getAll())

# categories_dt = np.dtype([('name', np.str, 10), ('categ', np.array)])
# exit()

categories = np.array([
    Category('road', road_categ),
    Category('pos', pos_categ),
    Category('1carAhead', one_car_ahead_approaching),
    Category('1carAheadRight', one_car_ahead_right_approaching),
    Category('1carAheadLeft', one_car_ahead_left_approaching),
    Category('surpassing', opponents_surpassing),
    Category('oppLeft', opponent_immediate_left),
    Category('oppRight', opponent_immediate_right),
    Category('countOppRight', count_opp_right),
    Category('countOppLeft', count_opp_left),
])

# print categories

tupleList = []

getInitialTuple = lambda: np.array(("", "", False, False, False, False, False, False, "0", "0"), dtype=tuple_dt)

# tuple = getInitialTuple()

# print getInitialTuple()
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

tuples_dt = np.dtype([
    ('id', np.int),
    ('tuple', tuple_dt)
])
tuples = np.array([(i, tuple) for i, tuple in enumerate(tupleList)], dtype=tuples_dt)

# clean the surpassing ones
# toDelete = np.array([(
#                          (x['surpassing'] == True and x['1carAheadLeft'] == True) or
#                          (x['surpassing'] == True and x['1carAheadRight'] == True) or
#                          (x['surpassing'] == True and x['1carAhead'] == True)
#                      ) for x in [t['tuple'] for t in tuples]])
#states = tuples[toDelete == False]
states = tuples[:]

print len(states)
print states[0]

filename = os.path.splitext(os.path.basename(__file__))[0]
np.save(filename + '.npy', states)
