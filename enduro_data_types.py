import numpy as np
from road_category import RoadCategory
from extreme_position import ExtremePosition

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