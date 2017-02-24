import os
import numpy as np
from road_category import RoadCategory
from extreme_position import ExtremePosition
from enduro_data_types import tuple_boeing_dt
from category import Category
from state_generator import generateStates
from how_many_opponents_discrete import HowManyOpponentsDiscrete

# for k, v in np.sctypeDict.iteritems(): print '{0:14s} : {1:40s}'.format(str(k), v)

categories = np.array([
    Category(
        'road', np.array(RoadCategory().getAll())
    ),
    Category(
        'pos', np.array(ExtremePosition().getAll())
    ),
    Category(
        'oppNearLeft', np.array(HowManyOpponentsDiscrete().getAll())
    ),
    Category(
        'oppNearRight', np.array(HowManyOpponentsDiscrete().getAll())
    ),
    Category(
        'oppFarLeft', np.array(HowManyOpponentsDiscrete().getAll())
    ),
    Category(
        'oppFarRight', np.array(HowManyOpponentsDiscrete().getAll())
    ),
    Category(
        'surpassing', np.array([False, True])
    ),
    Category(
        'oppLeft', np.array([False, True])
    ),
    Category(
        'oppRight', np.array([False, True])
    ),
])

# print categories

getInitialTuple = lambda: np.array(("", "", "", "", "", "", False, False, False), dtype=tuple_boeing_dt)

states = generateStates(categories, getInitialTuple, tuples_dt=np.dtype([
    ('id', np.int),
    ('tuple', tuple_boeing_dt)
])
                        )

filename = os.path.splitext(os.path.basename(__file__))[0]
np.save(filename + '.npy', states)

for i in range(len(states)):
    assert states[i]['id'] == i

print len(states)
print states[0]
