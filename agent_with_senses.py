from sense import Sense
import numpy as np
from enduro_data_types import tuple_dt

class AgentWithSenses(object):
    def __init__(self, rng):
        super(AgentWithSenses, self).__init__()
        # Add member variables to your class here
        self.sensor = Sense(rng)

    def getStateIds(self):
        return [s['id'] for s in self.states]

    def getStateIdBySensing(self, prevGrid, action, newGrid):
        allSenses = self.getAllSenses(prevGrid, action, newGrid)
        # a = allSenses.copy()
        # a['road'] = "333"
        # print a == allSenses

        stateIds = [s['id'] for s in self.states if s['tuple'] == allSenses]

        assert len(stateIds) == 1
        stateId = stateIds[0]

        #print stateId
        return stateId

    def getAllSenses(self, prevGrid, action, newGrid):
        raise NotImplementedError()

    def getStateById(self, stateId):
        ss = [s for s in self.states if s['id'] == stateId]
        assert len(ss) == 1
        return ss[0]
