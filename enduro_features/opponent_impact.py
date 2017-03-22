from __future__ import division
import numpy as np
from enduro.action import Action
from sensor import Sensor
from feature_base import ContrainedFeature, Feature
from py_helper import Constrainer
from collections import OrderedDict

class OpponentImpactFeature(ContrainedFeature, Feature):
    def __init__(self, rng):
        super(OpponentImpactFeature, self).__init__()
        self.sensor = Sensor(rng)
        self.magnitude_constrainer = Constrainer(new_min=0, new_max=1)

        # self.sum_cos_sim = 0.
        # self.count_cos_sim = 0
        self.threshold_cos_sim = 0.5

    # def get_average_cos_sim(self, new_cos_sim):
    #     self.sum_cos_sim += new_cos_sim
    #     self.count_cos_sim += 1
    #     return self.sum_cos_sim / self.count_cos_sim

    def __getContrainedMagnitude(self, magnitude):
        self.magnitude_constrainer.old_range = magnitude
        return self.magnitude_constrainer.constrain(magnitude)

    def get_angle_magnitude(self, **kwargs):
        OPPONENT_INDEX = kwargs['OPPONENT_INDEX']

        cos_sim, isOpponentLeft, magnitude = self.sensor.getAngleAndMagnitudeOfOpponentFromEnv(
            cars=kwargs['cars'], road=kwargs['road'], opp_index=OPPONENT_INDEX
        )

        return cos_sim, isOpponentLeft, magnitude

    def getFeatureValue(self, cur_action, **kwargs):
        cos_sim = kwargs['cos_sim']
        magnitude = kwargs['magnitude']

        if cos_sim is None or magnitude is None:
            value = 0
        else:
            cos_sim_clipped = cos_sim if cos_sim > 0.1 else 0.1

            contrained_magnitude = self.__getContrainedMagnitude(magnitude=magnitude)

            # print "cur cos sim {} and average cos sim {}".format(
            #     cos_sim, self.get_average_cos_sim(new_cos_sim=cos_sim_clipped))

            inverse_cos_sim_clipped = 1 / cos_sim_clipped

            if cos_sim_clipped < self.threshold_cos_sim:
                value = inverse_cos_sim_clipped * contrained_magnitude
            else:
                value = inverse_cos_sim_clipped * (1. / contrained_magnitude)

        return super(OpponentImpactFeature, self).getFeatureValue(cur_action, value=value)


class FirstOpponentLeftFeature(OpponentImpactFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(FirstOpponentLeftFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['OPPONENT_INDEX'] = 0
        cos_sim, isOpponentLeft, magnitude = self.get_angle_magnitude(**kwargs)

        check = (isOpponentLeft is not None) and isOpponentLeft

        kwargs['cos_sim'] = cos_sim if check else None
        kwargs['magnitude'] = magnitude if check else None

        return super(FirstOpponentLeftFeature, self).getFeatureValue(cur_action, **kwargs)


class FirstOpponentRightFeature(OpponentImpactFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(FirstOpponentRightFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['OPPONENT_INDEX'] = 0
        cos_sim, isOpponentLeft, magnitude = self.get_angle_magnitude(**kwargs)

        check = (isOpponentLeft is not None) and (not isOpponentLeft)

        kwargs['cos_sim'] = cos_sim if check else None
        kwargs['magnitude'] = magnitude if check else None

        return super(FirstOpponentRightFeature, self).getFeatureValue(cur_action, **kwargs)


class SecondOpponentLeftFeature(OpponentImpactFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(SecondOpponentLeftFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['OPPONENT_INDEX'] = 1
        cos_sim, isOpponentLeft, magnitude = self.get_angle_magnitude(**kwargs)

        check = (isOpponentLeft is not None) and isOpponentLeft

        kwargs['cos_sim'] = cos_sim if check else None
        kwargs['magnitude'] = magnitude if check else None

        return super(SecondOpponentLeftFeature, self).getFeatureValue(cur_action, **kwargs)


class SecondOpponentRightFeature(OpponentImpactFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(SecondOpponentRightFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['OPPONENT_INDEX'] = 1
        cos_sim, isOpponentLeft, magnitude = self.get_angle_magnitude(**kwargs)

        check = (isOpponentLeft is not None) and (not isOpponentLeft)

        kwargs['cos_sim'] = cos_sim if check else None
        kwargs['magnitude'] = magnitude if check else None

        return super(SecondOpponentRightFeature, self).getFeatureValue(cur_action, **kwargs)


class ThirdOpponentLeftFeature(OpponentImpactFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(ThirdOpponentLeftFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['OPPONENT_INDEX'] = 2
        cos_sim, isOpponentLeft, magnitude = self.get_angle_magnitude(**kwargs)

        check = (isOpponentLeft is not None) and isOpponentLeft

        kwargs['cos_sim'] = cos_sim if check else None
        kwargs['magnitude'] = magnitude if check else None

        return super(ThirdOpponentLeftFeature, self).getFeatureValue(cur_action, **kwargs)


class ThirdOpponentRightFeature(OpponentImpactFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(ThirdOpponentRightFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['OPPONENT_INDEX'] = 1
        cos_sim, isOpponentLeft, magnitude = self.get_angle_magnitude(**kwargs)

        check = (isOpponentLeft is not None) and (not isOpponentLeft)

        kwargs['cos_sim'] = cos_sim if check else None
        kwargs['magnitude'] = magnitude if check else None

        return super(ThirdOpponentRightFeature, self).getFeatureValue(cur_action, **kwargs)
