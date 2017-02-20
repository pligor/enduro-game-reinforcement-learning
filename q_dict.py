from collections import OrderedDict
import numpy as np

class Qdict(object):
    def __init__(self):
        super(Qdict, self).__init__()
        assert hasattr(self, "states")
        assert hasattr(self, "getActionsSet")
        assert hasattr(self, "getStateIds")

        self.Qshape = (len(self.states), len(self.getActionsSet()))
        # self.bellmanQ = np.zeros(Qshape)
        # self.bellmanQ = OrderedDict(zip(self.getStateIds(), self.rng.rand(self.Qshape[0], self.Qshape[1])))
        self.bellmanQ = OrderedDict(zip(self.getStateIds(), np.zeros(self.Qshape)))

    def updateQsa(self, stateId, actionId, value):
        self.bellmanQ[stateId][actionId] = value
        return self.bellmanQ[stateId][actionId]

    def getQsa(self, stateId, actionId):
        return self.bellmanQ[stateId][actionId]

    def getQbyS(self, stateId):
        return self.bellmanQ[stateId]