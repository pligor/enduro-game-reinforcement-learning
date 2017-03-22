from __future__ import division
import numpy as np
from enduro.action import Action
from py_helper import Constrainer
from collections import OrderedDict


class WithFeatureValue(object):
    def getFeatureValue(self, cur_action, **kwargs):
        raise NotImplementedError


class WithGetPrior(object):
    def getPrior(self):
        raise NotImplementedError


class PlainFeature(WithFeatureValue, WithGetPrior):
    def __init__(self):
        super(PlainFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        return kwargs['value']

    def getPrior(self):
        if hasattr(self, "prior_weight"):
            return self.prior_weight
        else:
            raise AssertionError


class ConstantBiasPlainFeature(PlainFeature):
    def __init__(self, rng):
        self.prior_weight = 0.

        super(ConstantBiasPlainFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        return super(ConstantBiasPlainFeature, self).getFeatureValue(cur_action, value=1)


class Feature(WithFeatureValue, WithGetPrior):
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

        self.checkActionsOrder()

    def checkActionsOrder(self):
        if hasattr(self, "priors_per_action"):
            enabledActions = self.get_enabled_actions()
            actions = self.priors_per_action.keys()
            for action in actions:
                if action in enabledActions:
                    assert action == enabledActions[0], \
                        "enabled actions must have same order as priors, action {} and first en action {}".format(
                            action, enabledActions[0]
                        )
                    del enabledActions[0]

    @staticmethod
    def get_enabled_actions():
        return [Action.ACCELERATE, Action.RIGHT, Action.LEFT, Action.BRAKE, Action.NOOP]

    def getFeatureValue(self, cur_action, **kwargs):
        if hasattr(self, "corresponding_action"):
            return kwargs['value'] if cur_action == self.corresponding_action else 0
        else:
            raise AssertionError

    def getPrior(self):
        if hasattr(self, "priors_per_action") and hasattr(self, "corresponding_action"):
            return self.priors_per_action[self.corresponding_action]
        else:
            raise AssertionError


class ContrainedFeature(Constrainer, WithFeatureValue):
    def __init__(self, new_min=-1, new_max=+1):
        super(ContrainedFeature, self).__init__(new_min=new_min, new_max=new_max)

    def getFeatureValue(self, cur_action, **kwargs):
        cur_value = kwargs['value']
        self.old_range = cur_value

        # print "cur value {}".format(cur_value)

        constrained_value = self.constrain(cur_value)

        # print "constrained value {}".format(constrained_value)

        return super(ContrainedFeature, self).getFeatureValue(cur_action, value=constrained_value)


class ConstantBiasFeature(Feature):
    def __init__(self, corresponding_action, rng):
        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        self.corresponding_action = corresponding_action

        super(ConstantBiasFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        return super(ConstantBiasFeature, self).getFeatureValue(cur_action, value=1)
