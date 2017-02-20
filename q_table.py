import numpy as np

class Qtable(object):
    def __init__(self):
        super(Qtable, self).__init__()
        assert hasattr(self, "states")
        assert hasattr(self, "getActionsSet")

        self.Qshape = (len(self.states), len(self.getActionsSet()))
        # self.bellmanQ = np.zeros(Qshape)
        # self.bellmanQ = OrderedDict(zip(self.getStateIds(), self.rng.rand(self.Qshape[0], self.Qshape[1])))
        self.bellmanQ = np.zeros(self.Qshape)

    def updateQsa(self, stateId, actionId, value):
        self.bellmanQ[stateId, actionId] = value

    def getQsa(self, stateId, actionId):
        return self.bellmanQ[stateId, actionId]

    def getQbyS(self, stateId):
        return self.bellmanQ[stateId]