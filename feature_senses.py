from __future__ import division

import numpy as np
from how_many_opponents_discrete import HowManyOpponentsDiscrete
from sensor import Sensor
from enduro.action import Action
from collections import OrderedDict
from features_linear_approx import MoveFasterWhenLessThanAverageSpeed, MoveSlowerWhenMoreThanAverageSpeed
import features_linear_approx


class FeatureSenses(object):
    """['ACCELERATE', 'RIGHT', 'LEFT', 'BRAKE', 'NOOP']"""

    def __init__(self, rng):
        self.nonLinearitiesEnabled = False

        self.featureNameList = [
            'MoveFasterWhenLessThanAverageSpeed',
            'MoveSlowerWhenMoreThanAverageSpeed',
            'FirstOpponentFeature',
            'SecondOpponentFeature',
            'ThirdOpponentFeature'
        ]

        weight_priors = []
        for featureInstance in self.__featureIterator():
            weight_priors.append(
                featureInstance.getPriorForCorrespondingAction()
            )

        if hasattr(self, "getActionsSet"):
            action_set = self.getActionsSet()
        else:
            raise AssertionError

        originalFeatureLen = len(action_set) * len(self.featureNameList)
        assert originalFeatureLen == len(weight_priors)

        # self.initialTheta = Qcase.RANDOM
        def changeWeights(vector):
            vector[:originalFeatureLen] = weight_priors
            #fill the rest with random values
            vector[vector == 0] = rng.randn(len(vector[vector == 0]))
            return vector

        self.initialTheta = changeWeights

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

    def stayingInTheCentreOfTheRoad(self, grid, action, road):
        return (
            self.sensor.distanceFromCentre(grid=grid, action=action),
            self.sensor.opponentsBeside(grid=grid, action=action),
            self.sensor.howMuchRoadTurning(road=road, action=action)
        )

    def __featureIterator(self):
        if hasattr(self, "getActionsSet") and hasattr(self, "rng"):
            action_set = self.getActionsSet()

            featureInstances = []

            for cur_feat_name in self.featureNameList:
                for cur_action in action_set:
                    featureInstance = getattr(features_linear_approx, cur_feat_name)(
                        corresponding_action=cur_action, rng=self.rng
                    )

                    featureInstances.append(featureInstance)

            return featureInstances
        else:
            raise AssertionError

    def __getFeatureVector(self, prevEnv, action, curEnv):
        if prevEnv is None:
            return np.zeros(self.featureLen)
        else:
            # according to theory the state corresponds to the grid AFTER the action is taken
            # (distanceFromCentre, opponentsBeside, howMuchRoadTurning) = self.stayingInTheCentreOfTheRoad(
            #     grid=curEnv['grid'],
            #     road=curEnv['road'],
            #     action=action
            # )

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

            featureValuesVector = []

            for featureInstance in self.__featureIterator():
                featureValuesVector.append(
                    featureInstance.getFeatureValue(cur_action=action,
                                                    speed=curEnv['speed'],
                                                    cars=curEnv['cars'],
                                                    road=curEnv['road'])
                )

            featureValuesVector = np.array(featureValuesVector)

            if self.nonLinearitiesEnabled:
                featureValuesVector = np.concatenate(
                    (featureValuesVector, self.multiplyCombinations(featureValuesVector)))
                assert len(featureValuesVector) == self.featureLen

            return featureValuesVector

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
