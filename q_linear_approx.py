import numpy as np

from q_case import Qcase

class Q_LinearApprox(object):
    def __init__(self):
        super(Q_LinearApprox, self).__init__()

        if hasattr(self, "featureLen"):
            if hasattr(self, "initialTheta"):
                if self.initialTheta == Qcase.ZERO:
                    self.thetaVector = np.zeros(self.featureLen)

                elif self.initialTheta == Qcase.RANDOM:
                    if hasattr(self, "rng"):
                        self.thetaVector = self.rng.rand(self.featureLen)
                    else:
                        raise AssertionError

                else:
                    self.thetaVector = self.initialTheta(
                        np.zeros(self.featureLen)
                    )
            else:
                self.thetaVector = np.zeros(self.featureLen)
        else:
            raise AssertionError

        if hasattr(self, "getActionsSet"):
            self._actionsLen = len(self.getActionsSet())
        else:
            raise AssertionError

        self._thetaVecLen = len(self.thetaVector)

    def updateQsa(self, learningRate, curReward, gamma, curFeatureVectorsForAllActions, prevFeatureVector):
        assert learningRate > 0

        self.thetaVector += learningRate * (
            curReward + gamma * np.max(self.getQbyS(curFeatureVectorsForAllActions)) - self.getQsa(prevFeatureVector)
        ) * prevFeatureVector

    def getQsa(self, featureVector):
        return self.thetaVector.dot(featureVector)

    def getQbyS(self, featureVectorsForAllActions):
        """each column is a feature vector for a particular action"""
        assert featureVectorsForAllActions.shape == (self._thetaVecLen, self._actionsLen)

        return self.thetaVector.dot(featureVectorsForAllActions)