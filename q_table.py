import numpy as np

class Qcase(object):
    RANDOM = "random"
    ZERO = "zero"

    def __init__(self):
        super(Qcase, self).__init__()


class Qtable(object):
    def __init__(self):
        super(Qtable, self).__init__()
        assert hasattr(self, "states")
        assert hasattr(self, "getActionsSet")
        assert hasattr(self, "rng")

        qshape = self.getQshape()
        # self.bellmanQ = OrderedDict(zip(self.getStateIds(), self.rng.rand(self.Qshape[0], self.Qshape[1])))

        if (not hasattr(self, "initialQ")) or self.initialQ == Qcase.ZERO:
            self.bellmanQ = np.zeros(qshape)
        elif self.initialQ == Qcase.RANDOM:
            self.bellmanQ = self.rng.rand(qshape[0], qshape[1])
        else:
            self.bellmanQ = self.initialQ(np.zeros(qshape))

    def getQshape(self):
        return (len(self.states), len(self.getActionsSet()))

    def updateQsa(self, stateId, actionId, value):
        self.bellmanQ[stateId, actionId] = value

    def getQsa(self, stateId, actionId):
        return self.bellmanQ[stateId, actionId]

    def getQbyS(self, stateId):
        return self.bellmanQ[stateId]