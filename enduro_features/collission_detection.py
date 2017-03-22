from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature, PlainFeature
from sensor import Sensor
from collections import OrderedDict


class ShortSightedOppViewFeature(Feature):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.sensor = Sensor(rng=rng)

        self.priors_per_action = OrderedDict([
            (Action.ACCELERATE, 1.),
            (Action.RIGHT, 5.),
            (Action.LEFT, 5.),
            (Action.BRAKE, 1.),
            (Action.NOOP, -5.),
        ])

        super(ShortSightedOppViewFeature, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        grid = kwargs['grid']

        carPos = self.sensor.getOurCarPos(grid=grid)
        goodness = self.sensor.senseOppsInFrontOfCar(grid=grid, carPos=carPos)

        return super(ShortSightedOppViewFeature, self).getFeatureValue(cur_action, value=goodness)
