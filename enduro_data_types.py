import numpy as np
from road_category import RoadCategory
from extreme_position import ExtremePosition
from how_many_opponents import HowManyOpponents
from how_many_opponents_discrete import HowManyOpponentsDiscrete

tuple_boeing_dt = np.dtype([
    ('road', np.str, np.max([len(x) for x in np.array(RoadCategory().getAll())])),
    ('pos', np.str, np.max([len(x) for x in np.array(ExtremePosition().getAll())])),
    ('oppNearLeft', np.str, np.max([len(x) for x in np.array(HowManyOpponentsDiscrete().getAll())])),
    ('oppNearRight', np.str, np.max([len(x) for x in np.array(HowManyOpponentsDiscrete().getAll())])),
    ('oppFarLeft', np.str, np.max([len(x) for x in np.array(HowManyOpponentsDiscrete().getAll())])),
    ('oppFarRight', np.str, np.max([len(x) for x in np.array(HowManyOpponentsDiscrete().getAll())])),
    ('surpassing', np.bool),
    ('oppLeft', np.bool),
    ('oppRight', np.bool),
])


tuple_v2_dt = np.dtype([
    ('pos', np.str, np.max([len(x) for x in np.array(ExtremePosition().getAll())])),
    ('oppLeft', np.str, np.max([len(x) for x in np.array(HowManyOpponentsDiscrete().getAll())])),
    ('oppRight', np.str, np.max([len(x) for x in np.array(HowManyOpponentsDiscrete().getAll())])),
    ('surpassing', np.bool),
])

tuple_dt = np.dtype([
    ('road', np.str, np.max([len(x) for x in np.array(RoadCategory().getAll())])),
    ('pos', np.str, np.max([len(x) for x in np.array(ExtremePosition().getAll())])),
    ('1carAhead', np.bool),
    ('1carAheadRight', np.bool),
    ('1carAheadLeft', np.bool),
    ('surpassing', np.bool),
    ('oppLeft', np.bool),
    ('oppRight', np.bool),
])

long_tuple_dt = np.dtype([
    ('road', np.str, np.max([len(x) for x in np.array(RoadCategory().getAll())])),
    ('pos', np.str, np.max([len(x) for x in np.array(ExtremePosition().getAll())])),
    ('1carAhead', np.bool),
    ('1carAheadRight', np.bool),
    ('1carAheadLeft', np.bool),
    ('surpassing', np.bool),
    ('oppLeft', np.bool),
    ('oppRight', np.bool),
    ('countOppRight', np.str, np.max([len(x) for x in np.array(HowManyOpponents().getAll())])),
    ('countOppLeft', np.str, np.max([len(x) for x in np.array(HowManyOpponents().getAll())])),
])

