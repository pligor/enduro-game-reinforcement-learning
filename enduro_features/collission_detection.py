from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature, PlainFeature
from sensor import Sensor
from collections import OrderedDict


class ReactToOppsDirectlyInFront(PlainFeature):
    def __init__(self, rng):
        self.prior_weight = 1.

        self.sensor = Sensor(rng=rng)

        super(ReactToOppsDirectlyInFront, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        grid = kwargs['grid']

        carPos = self.sensor.getOurCarPos(grid=grid)
        oppsCount = self.sensor.countOppsInRelationToCar(grid=grid, carPos=carPos, howFar=5)

        def getValue():
            if oppsCount > 0:
                if cur_action == Action.ACCELERATE:
                    return -1.
                elif cur_action == Action.BRAKE:
                    return 1.
                else:
                    return 0
            else:
                if cur_action == Action.ACCELERATE:
                    return 0.1
                elif cur_action == Action.BRAKE:
                    return -0.1
                else:
                    return 0

        value = getValue()

        #print "ReactToOppsDirectlyInFront {}".format(value)

        return super(ReactToOppsDirectlyInFront, self).getFeatureValue(cur_action, value=value)


class PenaltyIfCollissionFeature(Feature):
    min_speed = -50

    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.sensor = Sensor(rng=rng)

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 10.),
            (Action.RIGHT, 10.),
            (Action.LEFT, 10.),
            (Action.BRAKE, 10.),
            (Action.NOOP, 10.),
        ])

        super(PenaltyIfCollissionFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        speed = kwargs['speed']
        prevSpeed = kwargs['prevSpeed']

        if speed == self.min_speed and prevSpeed > self.min_speed + 10:
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
