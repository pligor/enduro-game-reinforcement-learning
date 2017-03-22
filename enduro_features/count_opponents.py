from __future__ import division
import numpy as np
from enduro.action import Action
from sensor import Sensor
from feature_base import ContrainedFeature, Feature
from collections import OrderedDict

# oppNearLeft = self.__countToTextClass(
#             self.sensor.countOppsVarLen(newGrid, left_boolean=True, howFar=5, startFrom=0))
#         oppNearRight = self.__countToTextClass(
#             self.sensor.countOppsVarLen(newGrid, left_boolean=False, howFar=5, startFrom=0))
#         oppFarLeft = self.__countToTextClass(
#             self.sensor.countOppsVarLen(newGrid, left_boolean=True, howFar=11, startFrom=6))
#         oppFarRight = self.__countToTextClass(
#             self.sensor.countOppsVarLen(newGrid, left_boolean=False, howFar=11, startFrom=6))

class CountOpponentsFeature(ContrainedFeature, Feature):
    def __init__(self, rng):
        super(CountOpponentsFeature, self).__init__()
        self.sensor = Sensor(rng)

    def getFeatureValue(self, cur_action, **kwargs):
        oppCount = self.sensor.countOppsVarLen(grid=kwargs['grid'],
                                               left_boolean=kwargs['left_boolean'],
                                               startFrom=kwargs['startFrom'],
                                               howFar=kwargs['howFar'])

        oppCount_clipped = oppCount if oppCount > 0 else 0.1

        inverse_opp_count_clipped = 1. / oppCount_clipped

        return super(CountOpponentsFeature, self).getFeatureValue(cur_action, value=inverse_opp_count_clipped)


class CountOppsNearLeftFeature(CountOpponentsFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(CountOppsNearLeftFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['left_boolean'] = True
        kwargs['startFrom'] = 2
        kwargs['howFar'] = 6

        return super(CountOppsNearLeftFeature, self).getFeatureValue(cur_action, **kwargs)


class CountOppsNearRightFeature(CountOpponentsFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(CountOppsNearRightFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['left_boolean'] = False
        kwargs['startFrom'] = 2
        kwargs['howFar'] = 6

        return super(CountOppsNearRightFeature, self).getFeatureValue(cur_action, **kwargs)


class CountOppsFarLeftFeature(CountOpponentsFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(CountOppsFarLeftFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['left_boolean'] = True
        kwargs['startFrom'] = 7
        kwargs['howFar'] = 10

        return super(CountOppsFarLeftFeature, self).getFeatureValue(cur_action, **kwargs)


class CountOppsFarRightFeature(CountOpponentsFeature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0),
            (Action.RIGHT, 0),
            (Action.LEFT, 0),
            (Action.BRAKE, 0),
            (Action.NOOP, 0),
        ])

        super(CountOppsFarRightFeature, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        kwargs['left_boolean'] = False
        kwargs['startFrom'] = 7
        kwargs['howFar'] = 10

        return super(CountOppsFarRightFeature, self).getFeatureValue(cur_action, **kwargs)
