from __future__ import division

import numpy as np
from how_many_opponents_discrete import HowManyOpponentsDiscrete
from sensor import Sensor
from enduro.action import Action
from collections import OrderedDict
from enduro_features.moving_faster import MoveFasterWhenLessThanAverageSpeed, MoveSlowerWhenMoreThanAverageSpeed
from enduro_features.distance_centre import MoveLeftWhenRight, MoveRightWhenLeft


class FeatureSenses(object):
    """['ACCELERATE', 'RIGHT', 'LEFT', 'BRAKE', 'NOOP']"""

    def __init__(self, rng):
        self.nonLinearitiesEnabled = False

        feature_class_list = [
            MoveFasterWhenLessThanAverageSpeed,
            MoveSlowerWhenMoreThanAverageSpeed,
            MoveLeftWhenRight,
            MoveRightWhenLeft
            # 'FirstOpponentFeature',
            # 'SecondOpponentFeature',
            # 'ThirdOpponentFeature',
            # 'ConstantBiasFeature'
        ]

        weight_priors = []

        self.featureList = self.__generateFeatures(feature_class_list=feature_class_list)

        for featureInstance in self.featureList:
            weight_priors.append(
                featureInstance.getPriorForCorrespondingAction()
            )

        if hasattr(self, "getActionsSet"):
            action_set = self.getActionsSet()
        else:
            raise AssertionError

        originalFeatureLen = len(action_set) * len(feature_class_list)
        assert originalFeatureLen == len(weight_priors)

        # self.initialTheta = Qcase.RANDOM
        def changeWeights(vector):
            vector[:originalFeatureLen] = weight_priors
            # fill the rest with random values
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
            self.sensor.opponentsBeside(grid=grid, action=action),
            self.sensor.howMuchRoadTurning(road=road, action=action)
        )

    def __generateFeatures(self, feature_class_list):
        if hasattr(self, "getActionsSet") and hasattr(self, "rng"):
            action_set = self.getActionsSet()

            featureInstances = []

            for cur_feat_class in feature_class_list:
                for cur_action in action_set:
                    featureInstance = cur_feat_class(
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
            cars = curEnv['cars']
            road = np.array(curEnv['road'])
            grid = curEnv['grid']

            def loop_road(coords_opp):
                """the other idea is to split it in four sections and the winner section to even four"""
                prev_row = None
                for rr, row in enumerate(road):
                    prev_col = None  # (0,0)
                    for cc, col in enumerate(row):
                        # (col[1] > coords_opp[1] > prev_col[1]):
                        if prev_row is not None and prev_col is not None:

                            # print (prev_row[cc][0], col[0])
                            if (prev_col[0] <= coords_opp[0] <= col[0]) and \
                                    (prev_row[cc][1] <= coords_opp[1] <= col[1]):
                                return rr, cc
                        #
                        # if prev_col is None:
                        #
                        #     pass
                        # else:
                        #     if coords_opp[0] > col[0] and prev_col[1] < coords_opp[1] < col[1]:
                        #         return rr, cc
                        prev_col = col
                    prev_row = row

            opponents = cars['others']

            if len(opponents) > 0:
                # print road[0]
                print grid

            for opp in opponents:
                coords_opponent = opp[:2]
                print coords_opponent
                tpl = loop_road(coords_opp=coords_opponent)
                xx = 11 - (tpl[0])
                print "loop road {}".format(tpl)
                #print grid[xx]
                min_xx = max(0, xx-1)
                max_xx = min(11, xx+2)
                #print xx, min_xx, max_xx
                #print grid[min_xx:max_xx]
                assert np.any(grid[min_xx:max_xx] == 1)
                #print road[tpl[0]]
                # print road[tpl[0] + 1]

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

            for featureInstance in self.featureList:
                featureValuesVector.append(
                    featureInstance.getFeatureValue(cur_action=action,
                                                    speed=curEnv['speed'],
                                                    cars=curEnv['cars'],
                                                    road=curEnv['road'],
                                                    grid=curEnv['grid'])
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
