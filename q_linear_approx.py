import numpy as np

class Qcase(object):
    RANDOM = "random"
    ZERO = "zero"

    def __init__(self):
        super(Qcase, self).__init__()


class Q_LinearApprox(object):
    def __init__(self):
        super(Q_LinearApprox, self).__init__()
        assert hasattr(self, "getActionsSet")
        assert hasattr(self, "rng")

        self.featureLen = 2 #TODO make it parametric

        if (not hasattr(self, "initialTheta")) or self.initialTheta == Qcase.ZERO:
            self.thetaVector = np.zeros(self.featureLen)

        elif self.initialTheta == Qcase.RANDOM:
            self.thetaVector = self.rng.rand(self.featureLen)

        else:
            self.thetaVector = self.initialTheta(
                np.zeros(self.featureLen)
            )

        self._actionsLen = len(self.getActionsSet())
        self._thetaVecLen = len(self.thetaVector)

    def updateQsa(self, learningRate, nextReward, gamma, nextFeatureVectors, curFeatureVector):
        assert learningRate > 0

        self.thetaVector += learningRate * (
            nextReward + gamma * np.max(self.getQbyS(nextFeatureVectors)) - self.getQsa(curFeatureVector)
        ) * curFeatureVector

    def getQsa(self, featureVector):
        return self.thetaVector.dot(featureVector)

    def getQbyS(self, featureVectorsForAllActions):
        """each column is a feature vector for a particular action"""
        assert featureVectorsForAllActions.shape == (self._thetaVecLen, self._actionsLen)

        return self.thetaVector.dot(featureVectorsForAllActions)