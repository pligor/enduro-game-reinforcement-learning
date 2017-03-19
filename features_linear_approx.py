from __future__ import division
import numpy as np
from enduro.action import Action
from sensor import Sensor


class WithFeatureValue(object):
    def getFeatureValue(self, cur_action, **kwargs):
        raise NotImplementedError


class Feature(WithFeatureValue):
    """['ACCELERATE', 'RIGHT', 'LEFT', 'BRAKE', 'NOOP']"""

    def __init__(self):
        super(Feature, self).__init__()
        if hasattr(self, "corresponding_action"):
            assert self.corresponding_action in (
                Action.ACCELERATE,
                Action.RIGHT,
                Action.LEFT,
                Action.BRAKE,
                Action.NOOP
            )
        else:
            raise AssertionError

        assert hasattr(self, "priors_per_action")

    def getFeatureValue(self, cur_action, **kwargs):
        if hasattr(self, "corresponding_action"):
            return kwargs['value'] if cur_action == self.corresponding_action else 0
        else:
            raise AssertionError

    def getPriorForCorrespondingAction(self):
        if hasattr(self, "priors_per_action") and hasattr(self, "corresponding_action"):
            return self.priors_per_action[self.corresponding_action]
        else:
            raise AssertionError


class ContrainedFeature(WithFeatureValue):
    def __init__(self, new_min=-1, new_max=+1):
        super(ContrainedFeature, self).__init__()
        self.new_min = new_min
        self.new_max = new_max
        self.new_range = self.new_max - self.new_min

        # start with the assumption that could be the same as our original ones
        self.old_min = new_min
        self.old_max = new_max

    @property
    def old_range(self):
        return self.old_max - self.old_min

    def constrain(self, old_value):
        """http://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio"""
        if self.old_range == 0:
            return (self.new_min + self.new_max) / 2
        else:
            return (((old_value - self.old_min) * self.new_range) / self.old_range) + self.new_min

    def getFeatureValue(self, cur_action, **kwargs):
        cur_value = kwargs['value']
        self.old_max = max(self.old_max, cur_value)
        self.old_min = min(self.old_min, cur_value)

        print "cur value {}".format(cur_value)

        constrained_value = self.constrain(cur_value)

        print "constrained value {}".format(constrained_value)

        return super(ContrainedFeature, self).getFeatureValue(cur_action, value=constrained_value)


class ConstantBiasFeature(Feature):
    def __init__(self, corresponding_action, rng):
        self.priors_per_action = {
            Action.ACCELERATE: 0,
            Action.RIGHT: 0,
            Action.LEFT: 0,
            Action.BRAKE: 0,
            Action.NOOP: 0,
        }

        self.corresponding_action = corresponding_action

        super(ConstantBiasFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        return super(ConstantBiasFeature, self).getFeatureValue(cur_action, value=1)


class OpponentImpactFeature(ContrainedFeature, Feature):
    def __init__(self, rng):
        super(OpponentImpactFeature, self).__init__()
        self.sensor = Sensor(rng)

    def getFeatureValue(self, cur_action, **kwargs):
        OPPONENT_INDEX = kwargs['OPPONENT_INDEX']

        # cos_sim, isOpponentLeft = self.sensor.getAngleOfOpponentFromEnv(
        #     cars=kwargs['cars'], road=kwargs['road'], opp_index=OPPONENT_INDEX
        # )
        cos_sim, isOpponentLeft, magnitude = self.sensor.getAngleAndMagnitudeOfOpponentFromEnv(
            cars=kwargs['cars'], road=kwargs['road'], opp_index=OPPONENT_INDEX
        )

        cos_sim_clipped = cos_sim if cos_sim > 0 else 0

        multiplier = -magnitude if isOpponentLeft else magnitude

        value = 0 if isOpponentLeft is None else cos_sim_clipped * multiplier

        return super(OpponentImpactFeature, self).getFeatureValue(cur_action, value=value)


class FirstOpponentFeature(OpponentImpactFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = {
            Action.ACCELERATE: 0,
            Action.RIGHT: 0,
            Action.LEFT: 0,
            Action.BRAKE: 0,
            Action.NOOP: 0,
        }

        super(FirstOpponentFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['OPPONENT_INDEX'] = 0
        return super(FirstOpponentFeature, self).getFeatureValue(cur_action, **kwargs)


class SecondOpponentFeature(OpponentImpactFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = {
            Action.ACCELERATE: 0,
            Action.RIGHT: 0,
            Action.LEFT: 0,
            Action.BRAKE: 0,
            Action.NOOP: 0,
        }

        super(SecondOpponentFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['OPPONENT_INDEX'] = 1
        return super(SecondOpponentFeature, self).getFeatureValue(cur_action, **kwargs)


class ThirdOpponentFeature(OpponentImpactFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = {
            Action.ACCELERATE: 0,
            Action.RIGHT: 0,
            Action.LEFT: 0,
            Action.BRAKE: 0,
            Action.NOOP: 0,
        }

        super(ThirdOpponentFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['OPPONENT_INDEX'] = 2
        return super(ThirdOpponentFeature, self).getFeatureValue(cur_action, **kwargs)


class MovingFasterResultsInPassingMoreCars(ContrainedFeature, Feature):
    default_rbf_wideness = 700
    default_average_speed = 30

    def __init__(self,
                 rbf_wideness=default_rbf_wideness, average_speed=default_average_speed):

        super(MovingFasterResultsInPassingMoreCars, self).__init__()
        self.average_speed = average_speed
        self.rbf_wideness = rbf_wideness

    def rbfFunc(self, x):
        return np.exp(-((x - self.average_speed) ** 2) / self.rbf_wideness)

    def getFeatureValue(self, cur_action, **kwargs):
        speed = kwargs['speed']
        speedFactor = self.rbfFunc(speed)
        return super(MovingFasterResultsInPassingMoreCars, self).getFeatureValue(cur_action, value=speedFactor)


class MoveFasterWhenLessThanAverageSpeed(MovingFasterResultsInPassingMoreCars):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = {
            Action.ACCELERATE: 9,
            Action.RIGHT: 1,
            Action.LEFT: 1,
            Action.BRAKE: -8,
            Action.NOOP: 1,
        }

        super(MoveFasterWhenLessThanAverageSpeed, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        isSlowerThanAverage = kwargs['speed'] < self.average_speed
        # return 0
        return super(MoveFasterWhenLessThanAverageSpeed, self).getFeatureValue(cur_action, **kwargs) \
            if isSlowerThanAverage else 0  # does not play role if


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
