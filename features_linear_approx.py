from __future__ import division
import numpy as np
from enduro.action import Action
from sensor import Sensor


class Feature(object):
    """['ACCELERATE', 'RIGHT', 'LEFT', 'BRAKE', 'NOOP']"""

    def __init__(self, corresponding_action):
        super(Feature, self).__init__()
        assert corresponding_action in (
            Action.ACCELERATE,
            Action.RIGHT,
            Action.LEFT,
            Action.BRAKE,
            Action.NOOP
        )

        assert hasattr(self, "priors_per_action")

        self.corresponding_action = corresponding_action

    def getFeatureValue(self, cur_action, **kwargs):
        return kwargs['value'] if cur_action == self.corresponding_action else 0

    def getPriorForCorrespondingAction(self):
        if hasattr(self, "priors_per_action"):
            return self.priors_per_action[self.corresponding_action]
        else:
            raise AssertionError


class MovingFasterResultsInPassingMoreCars(Feature):
    default_rbf_wideness = 700
    default_average_speed = 20

    def __init__(self, corresponding_action,
                 rbf_wideness=default_rbf_wideness, average_speed=default_average_speed):
        super(MovingFasterResultsInPassingMoreCars, self).__init__(corresponding_action=corresponding_action)
        self.average_speed = average_speed
        self.rbf_wideness = rbf_wideness

    def rbfFunc(self, x):
        return np.exp(-((x - self.average_speed) ** 2) / self.rbf_wideness)

    def getFeatureValue(self, cur_action, **kwargs):
        speed = kwargs['speed']
        speedFactor = self.rbfFunc(speed)
        return super(MovingFasterResultsInPassingMoreCars, self).getFeatureValue(cur_action, value=speedFactor)


class MoveFasterWhenLessThanAverageSpeed(MovingFasterResultsInPassingMoreCars):
    def __init__(self, corresponding_action):
        self.priors_per_action = {
            Action.ACCELERATE: 9,
            Action.RIGHT: 1,
            Action.LEFT: 1,
            Action.BRAKE: -8,
            Action.NOOP: 1,
        }

        super(MoveFasterWhenLessThanAverageSpeed, self).__init__(corresponding_action=corresponding_action)

    def getFeatureValue(self, cur_action, **kwargs):
        isSlowerThanAverage = kwargs['speed'] < self.average_speed
        # return 0
        return super(MoveFasterWhenLessThanAverageSpeed, self).getFeatureValue(cur_action, **kwargs) \
            if isSlowerThanAverage else 0  # does not play role if


class MoveSlowerWhenMoreThanAverageSpeed(MovingFasterResultsInPassingMoreCars):
    def __init__(self, corresponding_action):
        self.priors_per_action = {
            Action.ACCELERATE: -8,
            Action.RIGHT: 1,
            Action.LEFT: 1,
            Action.BRAKE: 9,
            Action.NOOP: 1,
        }

        super(MoveSlowerWhenMoreThanAverageSpeed, self).__init__(corresponding_action=corresponding_action)

    def getFeatureValue(self, cur_action, **kwargs):
        isFasterThanAverage = kwargs['speed'] > self.average_speed
        # return 0
        return super(MoveSlowerWhenMoreThanAverageSpeed, self).getFeatureValue(cur_action, **kwargs) \
            if isFasterThanAverage else 0  # does not play role if
