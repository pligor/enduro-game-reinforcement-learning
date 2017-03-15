import numpy as np
from how_many_opponents_discrete import HowManyOpponentsDiscrete
from sensor import Sensor


class FeatureSenses(object):
    def __init__(self, rng):
        self.nonLinearitiesEnabled = False

        originalFeatureLen = 2
        if self.nonLinearitiesEnabled:
            self.featureLen = originalFeatureLen + self.countCombinations(originalFeatureLen=originalFeatureLen)
        else:
            self.featureLen = originalFeatureLen

        super(FeatureSenses, self).__init__()
        self.sensor = Sensor(rng)

    def getFeatureVectorsForAllActions(self, prevEnv, curEnv):
        if hasattr(self, "getActionsSet"):
            return np.array(
                [self.__getFeatureVector(prevEnv=prevEnv, action=curAction, curEnv=curEnv) for curAction in
                 self.getActionsSet()]).T
        else:
            raise AssertionError

    def stayingInTheCentreOfTheRoad(self, grid, action):
        return (
            self.sensor.distanceFromCentre(grid, action),
            self.sensor.opponentsBeside(grid, action)
        )

    def __getFeatureVector(self, prevEnv, action, curEnv):
        if prevEnv is None:
            return np.zeros(self.featureLen)
        else:
            # according to theory the state corresponds to the grid AFTER the action is taken
            (distanceFromCentre, opponentsBeside) = self.stayingInTheCentreOfTheRoad(grid=curEnv['grid'],
                                                                                     action=action)
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
                opponentsBeside,
            ])

            if self.nonLinearitiesEnabled:
                featureVector = np.concatenate((featureVector, self.multiplyCombinations(featureVector)))
                assert len(featureVector) == self.featureLen

            return featureVector

    @staticmethod
    def countCombinations(originalFeatureLen):
        return ((originalFeatureLen ** 2) / 2) - (originalFeatureLen / 2)

    @staticmethod
    def multiplyCombinations(arr):
        arr = np.array(arr)  # make sure it is numpy array
        combs = FeatureSenses.takeCombinations(len(arr))

        products = []
        for comb in combs:
            product = 1
            for elem in arr[comb]:
                product *= elem
            products.append(product)

        return np.array(products)

    @staticmethod
    def takeCombinations(arrLen):
        """
        usage:
        arr = np.array([1, 2, 3])

        combs = takeCombinations(arr)
        for comb in combs:
            elems = arr[comb]
            print elems

            product = 1
            for elem in elems:
                product *= elem
            print product
        :param arrLen:
        :return:
        """
        inds = range(arrLen)
        couples = np.array(
            [tuple(np.sort(tpl)) for tpl in np.array(np.meshgrid(inds, inds)).T.reshape(-1, 2) if tpl[0] != tpl[1]])
        combinations = np.vstack({tuple(row) for row in couples})
        return combinations

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
