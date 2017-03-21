from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature, PlainFeature
from sensor import Sensor

class GoOrBrakePlainFeature(PlainFeature):
    def __init__(self, rng):
        self.prior_weight = 1.

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


class MovingFasterResultsInPassingMoreCars(Feature):
    default_rbf_wideness = 1200  # 700
    default_average_speed = 50  # 40, 30

    def __init__(self,
                 rbf_wideness=default_rbf_wideness, average_speed=default_average_speed):
        super(MovingFasterResultsInPassingMoreCars, self).__init__()
        self.average_speed = average_speed
        self.rbf_wideness = rbf_wideness

    def rbfFunc(self, x):
        value = np.exp(-((x - self.average_speed) ** 2) / self.rbf_wideness)
        assert value < 1.01
        return value

    def getFeatureValue(self, cur_action, **kwargs):
        speed = kwargs['speed']
        speedFactor = self.rbfFunc(speed)
        return super(MovingFasterResultsInPassingMoreCars, self).getFeatureValue(cur_action, value=speedFactor)


class MoveFasterWhenLessThanAverageSpeed(MovingFasterResultsInPassingMoreCars):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = {
            Action.ACCELERATE: 0.9,
            Action.RIGHT: 0.1,
            Action.LEFT: 0.1,
            Action.BRAKE: -0.8,
            Action.NOOP: 0.1,
        }

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

        self.priors_per_action = {
            Action.ACCELERATE: -8,
            Action.RIGHT: 1,
            Action.LEFT: 1,
            Action.BRAKE: 9,
            Action.NOOP: 1,
        }

        super(MoveSlowerWhenMoreThanAverageSpeed, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        isFasterThanAverage = kwargs['speed'] > self.average_speed
        # return 0
        return super(MoveSlowerWhenMoreThanAverageSpeed, self).getFeatureValue(cur_action, **kwargs) \
            if isFasterThanAverage else 0  # does not play role if
