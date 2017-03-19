from __future__ import division
import numpy as np
from enduro.action import Action
from sensor import Sensor
from feature_base import ContrainedFeature, Feature


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
