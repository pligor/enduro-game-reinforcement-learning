from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature, PlainFeature
from sensor import Sensor
from collections import OrderedDict


class WithRbfFunc(object):  # We have verified that rbf func always returns below 1 so ok
    @property
    def average_speed(self):
        raise NotImplementedError

    @property
    def rbf_wideness(self):
        raise NotImplementedError

    def rbfFunc(self, x):
        value = np.exp(-((x - self.average_speed) ** 2) / self.rbf_wideness)
        # assert value < 1.01
        return value


class GoOrBrakePlainFeature(PlainFeature):
    def __init__(self, rng):
        self.prior_weight = 10.

        self.sensor = Sensor(rng=rng)
        self.how_far = 10

        super(GoOrBrakePlainFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        grid = kwargs['grid']

        carPos = self.sensor.getOurCarPos(grid=grid)
        oppsCount = self.sensor.countOppsInFrontOfCar(grid=grid, carPos=carPos, howFar=self.how_far)

        if cur_action == Action.ACCELERATE and oppsCount == 0:
            value = 10
        elif cur_action == Action.ACCELERATE and oppsCount > 0:
            value = -1
        elif cur_action == Action.BRAKE and oppsCount > 0:
            value = 1
        elif cur_action == Action.BRAKE and oppsCount == 0:
            value = -10
        else:
            value = 0

        return super(GoOrBrakePlainFeature, self).getFeatureValue(cur_action, value=value)


class MovingFasterIsBetterPlainFeature(PlainFeature, WithRbfFunc):
    default_rbf_wideness = 1200  # 700
    default_average_speed = 50  # 40, 30
    max_speed = 50

    @property
    def average_speed(self):
        return self.default_average_speed

    @property
    def rbf_wideness(self):
        return self.default_rbf_wideness

    def __init__(self, rng):
        self.prior_weight = 10.

        super(MovingFasterIsBetterPlainFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        speed = kwargs['speed']
        speedFactor = self.rbfFunc(speed)

        def getValue():
            if cur_action == Action.ACCELERATE and speed < self.max_speed:
                return speedFactor
            elif cur_action == Action.ACCELERATE and speed == self.max_speed:
                return 0
            elif cur_action == Action.NOOP and speed < self.max_speed:
                return speedFactor / 10.
            elif cur_action == Action.NOOP and speed == self.max_speed:
                return 1
            elif cur_action == Action.BRAKE:
                return -1. / speedFactor
            else:
                return 0

        return super(MovingFasterIsBetterPlainFeature, self).getFeatureValue(cur_action, value=getValue())


class MovingFasterResultsInPassingMoreCars(Feature, WithRbfFunc):
    default_rbf_wideness = 1200  # 700
    default_average_speed = 50  # 40, 30

    @property
    def average_speed(self):
        return self.default_average_speed

    @property
    def rbf_wideness(self):
        return self.default_rbf_wideness

    @staticmethod
    def get_enabled_actions():
        return [Action.ACCELERATE, Action.BRAKE, Action.NOOP]

    def __init__(self):
        super(MovingFasterResultsInPassingMoreCars, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        speed = kwargs['speed']
        speedFactor = self.rbfFunc(speed)
        return super(MovingFasterResultsInPassingMoreCars, self).getFeatureValue(cur_action, value=speedFactor)


class MoveFasterWhenLessThanAverageSpeed(MovingFasterResultsInPassingMoreCars):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0.9),
            (Action.RIGHT, 0.1),
            (Action.LEFT, 0.1),
            (Action.BRAKE, -0.8),
            (Action.NOOP, 0.1),
        ])

        super(MoveFasterWhenLessThanAverageSpeed, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        isSlowerThanAverage = kwargs['speed'] <= self.average_speed

        qValue = super(MoveFasterWhenLessThanAverageSpeed, self).getFeatureValue(cur_action, **kwargs) \
            if isSlowerThanAverage else 0

        if self.average_speed >= 50 and self.corresponding_action == cur_action:
            assert qValue != 0, "q value is: {} and boolean: {} and speed: {}".format(
                qValue, isSlowerThanAverage, kwargs['speed'])

        return qValue


class MoveSlowerWhenMoreThanAverageSpeed(MovingFasterResultsInPassingMoreCars):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, -8),
            (Action.RIGHT, 1),
            (Action.LEFT, 1),
            (Action.BRAKE, 9),
            (Action.NOOP, 1),
        ])

        super(MoveSlowerWhenMoreThanAverageSpeed, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        isFasterThanAverage = kwargs['speed'] > self.average_speed
        # return 0
        return super(MoveSlowerWhenMoreThanAverageSpeed, self).getFeatureValue(cur_action, **kwargs) \
            if isFasterThanAverage else 0  # does not play role if
