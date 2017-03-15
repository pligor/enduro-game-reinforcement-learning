import numpy as np
from how_many_opponents_discrete import HowManyOpponentsDiscrete
from sensor import Sensor


class FeatureSenses(object):
    def __init__(self, rng):
        self.featureLen = 1

        super(FeatureSenses, self).__init__()
        self.sensor = Sensor(rng)

    def getFeatureVectorsForAllActions(self, prevGrid, curGrid):
        if hasattr(self, "getActionsSet"):
            return np.array(
                [self.getFeatureVector(prevGrid=prevGrid, action=curAction, curGrid=curGrid) for curAction in
                 self.getActionsSet()]).T
        else:
            raise AssertionError

    def stayingInTheCentreOfTheRoad(self, grid, action):
        # detect if opponents are found on the left and the action is left then this should be a small value
        # if opponents are found on the right and the action is right then small value
        # break should have a smaller impact, noop smaller
        # accelerate is preferred, the opposite might be also preferred

        # road is turning left by some degree, or right. If road is turning left, left action should be
        # preferred. The rest in that order: accelerate, noop, break, right
        # the same for if road is turning right
        return (self.sensor.distanceFromCentre(grid, action), )

    def getFeatureVector(self, prevGrid, action, curGrid):
        if prevGrid is None:
            return np.zeros(self.featureLen)
        else:
            (distanceFromCentre,) = self.stayingInTheCentreOfTheRoad(grid=curGrid,
                                                                     action=action)  # TODO or prevGrid, not sure..
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

            featureVector = np.array([
                distanceFromCentre,
            ])

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
