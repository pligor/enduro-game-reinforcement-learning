from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature, PlainFeature
from sensor import Sensor
from collections import OrderedDict


class PenaltyIfCollissionFeature(Feature):
    min_speed = -50

    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.sensor = Sensor(rng=rng)

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0.),
            (Action.RIGHT, 0.),
            (Action.LEFT, 0.),
            (Action.BRAKE, 1.),
            (Action.NOOP, 0.),
        ])

        super(PenaltyIfCollissionFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        speed = kwargs['speed']
        prevSpeed = kwargs['prevSpeed']

        if speed == self.min_speed and prevSpeed > 0:
            value = -1.
        else:
            value = 0

        return super(PenaltyIfCollissionFeature, self).getFeatureValue(cur_action, value=value)


class ShortSightedOppViewFeature(ContrainedFeature, Feature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.sensor = Sensor(rng=rng)

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 0.1),
            (Action.RIGHT, 0.5),
            (Action.LEFT, 0.5),
            (Action.BRAKE, 0.1),
            (Action.NOOP, -0.5),
        ])

        super(ShortSightedOppViewFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        grid = kwargs['grid']

        carPos = self.sensor.getOurCarPos(grid=grid)
        goodness = self.sensor.senseOppsInFrontOfCar(grid=grid, carPos=carPos, factor=2.)

        # print "goodness {}".format(goodness)

        return super(ShortSightedOppViewFeature, self).getFeatureValue(cur_action, value=goodness)
