import os
import numpy as np
from extreme_position import ExtremePosition
from enduro_data_types import tuple_v2_dt
from category import Category
from state_generator import generateStates
from how_many_opponents_discrete import HowManyOpponentsDiscrete

pos_categ = np.array(ExtremePosition().getAll())
oppLeft = np.array(HowManyOpponentsDiscrete().getAll())
oppRight = np.array(HowManyOpponentsDiscrete().getAll())
surpassing = np.array([False, True])

categories = np.array([
    Category('pos', pos_categ),
    Category('oppLeft', oppLeft),
    Category('oppRight', oppRight),
    Category('surpassing', surpassing),
])

getInitialTuple = lambda: np.array(("", "", "", False), dtype=tuple_v2_dt)

states = generateStates(categories, getInitialTuple, tuples_dt = np.dtype([
        ('id', np.int),
        ('tuple', tuple_v2_dt)
    ])
)

filename = os.path.splitext(os.path.basename(__file__))[0]
np.save(filename + '.npy', states)

for i in range(len(states)):
    assert states[i]['id'] == i

print len(states)
print states[0]
