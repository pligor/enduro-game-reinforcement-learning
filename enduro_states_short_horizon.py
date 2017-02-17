import numpy as np
from road_category import RoadCategory
from extreme_position import ExtremePosition

# for k, v in np.sctypeDict.iteritems(): print '{0:14s} : {1:40s}'.format(str(k), v)
# exit()

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
three_cars_one_step_back_approaching = np.array([False, True])
opponents_surpassing = np.array([False, True])
opponent_immediate_left = np.array([False, True])
opponent_immediate_right = np.array([False, True])

# categories_dt = np.dtype([('name', np.str, 10), ('categ', np.array)])
# exit()

categories = np.array([
    Category('road', road_categ),
    Category('pos', pos_categ),
    Category('1carAhead', one_car_ahead_approaching),
    Category('1carAheadRight', one_car_ahead_right_approaching),
    Category('1carAheadLeft', one_car_ahead_left_approaching),
    Category('3cars', three_cars_one_step_back_approaching),
    Category('surpassing', opponents_surpassing),
    Category('oppLeft', opponent_immediate_left),
    Category('oppRight', opponent_immediate_right),
])

# print categories

tuple_dt = np.dtype([
    ('road', np.str, np.max([len(x) for x in road_categ])),
    ('pos', np.str, np.max([len(x) for x in pos_categ])),
    ('1carAhead', np.bool),
    ('1carAheadRight', np.bool),
    ('1carAheadLeft', np.bool),
    ('3cars', np.bool),
    ('surpassing', np.bool),
    ('oppLeft', np.bool),
    ('oppRight', np.bool),
])

tupleList = []

getInitialTuple = lambda: np.array(("", "", False, False, False, False, False, False, False), dtype=tuple_dt)


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
toDelete = np.array([(
                         (x['surpassing'] == True and x['3cars'] == True) or
                         (x['surpassing'] == True and x['1carAheadLeft'] == True) or
                         (x['surpassing'] == True and x['1carAheadRight'] == True) or
                         (x['surpassing'] == True and x['1carAhead'] == True)
                     ) for x in [t['tuple'] for t in tuples]])

# print toDelete

# for t in tuples:
# np.argwhere( )

# print len(tuples)
# print tuples

states = tuples[toDelete == False]
# print len(toDelete[toDelete==True])

print len(states)
print states[0]

np.save('enduro_short_orizon_states.npy', states)
