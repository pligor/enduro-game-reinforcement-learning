from __future__ import division
import numpy as np
from enduro.action import Action

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

        #print "cur value {}".format(cur_value)

        constrained_value = self.constrain(cur_value)

        #print "constrained value {}".format(constrained_value)

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
