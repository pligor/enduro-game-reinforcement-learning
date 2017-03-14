import numpy as np
from how_many_opponents_discrete import HowManyOpponentsDiscrete
from sense import Sense


class FeatureSenses(object):
    def __init__(self, rng):
        self.featureLen = 2

        super(FeatureSenses, self).__init__()
        self.sensor = Sense(rng)

    def getFeatureVectorsForAllActions(self, prevGrid, curGrid):
        if hasattr(self, "getActionsSet"):
            return np.array(
                [self.getFeatureVector(prevGrid=prevGrid, action=curAction, newGrid=curGrid) for curAction in
                 self.getActionsSet()]).T
        else:
            raise AssertionError

    def getFeatureVector(self, prevGrid, action, newGrid):
        if prevGrid is None:
            return np.zeros(self.featureLen)
        else:
            # roadCateg = self.sensor.getRoadCateg(prevGrid, action, newGrid)
            # extremePos = self.sensor.getExtremePosition(latestGrid=newGrid)
            #
            # oppNearLeft = self.__countToTextClass(
            #     self.sensor.countOppsVarLen(newGrid, left_boolean=True, howFar=5, startFrom=0))
            # oppNearRight = self.__countToTextClass(
            #     self.sensor.countOppsVarLen(newGrid, left_boolean=False, howFar=5, startFrom=0))
            # oppFarLeft = self.__countToTextClass(
            #     self.sensor.countOppsVarLen(newGrid, left_boolean=True, howFar=11, startFrom=6))
            # oppFarRight = self.__countToTextClass(
            #     self.sensor.countOppsVarLen(newGrid, left_boolean=False, howFar=11, startFrom=6))
            #
            # areOpponentsSurpassing = self.sensor.doesOpponentSurpasses(prevGrid, newGrid)
            # isOpponentAtImmediateLeft = self.sensor.isOpponentAtImmediate(newGrid, right_boolean=False)
            # isOpponentAtImmediateRight = self.sensor.isOpponentAtImmediate(newGrid, right_boolean=True)
            #
            # featureVector = np.array((roadCateg,
            #                           extremePos,
            #                           oppNearLeft,
            #                           oppNearRight,
            #                           oppFarLeft,
            #                           oppFarRight,
            #                           areOpponentsSurpassing,
            #                           isOpponentAtImmediateLeft,
            #                           isOpponentAtImmediateRight))

            featureVector = np.array([1.4, 1.1])

            return featureVector

    @staticmethod
    def __countToTextClass(count):
        count = int(count)
        assert count >= 0
        if count == 0:
            return HowManyOpponentsDiscrete().none
        elif 1 <= count <= 2:
            return HowManyOpponentsDiscrete().afew
        else:
            return HowManyOpponentsDiscrete().alot
