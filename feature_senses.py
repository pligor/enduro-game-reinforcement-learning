from __future__ import division

import numpy as np
from how_many_opponents_discrete import HowManyOpponentsDiscrete
from sensor import Sensor
from enduro.action import Action


class FeatureSenses(object):
    def __init__(self, rng):
        self.nonLinearitiesEnabled = False

        originalFeatureLen = 4
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

    def collisionsShouldBeAvoided(self):
        # speed dependent, for high speeds
        # road dependent, turning left or right
        # use cars positions and predict next move

        # if road is turning right then car position is going to move diagonally
        # are we within the predicted range? If yes then we should turn outside of the range
        # if all cars block us then we should brake
        # the higher the speed the higher the need to turn and the further away we look at the road
        return

    @staticmethod
    def movingFasterResultsInPassingMoreCars(speed, action,
                                             averageSpeed=20,
                                             rbfSmoothness = 700,
                                             factors = (0.1, 0.5, 0.9)):
        # Consider speed itself to give some value, for example low speeds are really bad, but also high speeds are bad
        # (not as bad though), and of course differentiate per action

        rbfFunc = lambda x: np.exp(-((x - averageSpeed) ** 2) / rbfSmoothness)

        speedFactor = rbfFunc(speed)

        isSlowerThanAverage = speed < averageSpeed

        preferredAction = Action.ACCELERATE if isSlowerThanAverage else Action.BRAKE
        notPreferredAction = Action.BRAKE if isSlowerThanAverage else Action.ACCELERATE

        if action == preferredAction:
            return speedFactor * factors[2]
        elif action == Action.NOOP:
            return speedFactor * factors[1]
        elif action == notPreferredAction:
            return speedFactor * factors[0]
        else:
            return 0

    def stayingInTheCentreOfTheRoad(self, grid, action, road):
        return (
            self.sensor.distanceFromCentre(grid=grid, action=action),
            self.sensor.opponentsBeside(grid=grid, action=action),
            self.sensor.howMuchRoadTurning(road=road, action=action)
        )

    def __getFeatureVector(self, prevEnv, action, curEnv):
        if prevEnv is None:
            return np.zeros(self.featureLen)
        else:
            # according to theory the state corresponds to the grid AFTER the action is taken
            (distanceFromCentre, opponentsBeside, howMuchRoadTurning) = self.stayingInTheCentreOfTheRoad(
                grid=curEnv['grid'],
                road=curEnv['road'],
                action=action
            )

            howFastOurCarMoves = self.movingFasterResultsInPassingMoreCars(curEnv['speed'], action=action)
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

            featureVector = np.array([
                distanceFromCentre,
                opponentsBeside,
                howMuchRoadTurning,
                howFastOurCarMoves
            ])

            if self.nonLinearitiesEnabled:
                featureVector = np.concatenate((featureVector, self.multiplyCombinations(featureVector)))
                assert len(featureVector) == self.featureLen

            return featureVector

    @staticmethod
    def countCombinations(originalFeatureLen):
        return int(((originalFeatureLen ** 2) / 2) - (originalFeatureLen / 2))

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
