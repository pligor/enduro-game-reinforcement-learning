from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature, PlainFeature
from sensor import Sensor
from collections import OrderedDict

class SurpassNearbyOpponents(PlainFeature): #ContrainedFeature
    def __init__(self, rng):
        self.prior_weight = 1.

        self.sensor = Sensor(rng=rng)

        super(SurpassNearbyOpponents, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        grid = kwargs['grid']

        carPos = self.sensor.getOurCarPos(grid=grid)
        oppsCountLeft = self.sensor.countOppsInRelationToCar(grid=grid, carPos=carPos, howFar=5, shift=-3)
        oppsCountMiddle = self.sensor.countOppsInRelationToCar(grid=grid, carPos=carPos, howFar=5, shift=0)
        oppsCountRight = self.sensor.countOppsInRelationToCar(grid=grid, carPos=carPos, howFar=5, shift=+3)

        def getValue():
            if cur_action == Action.ACCELERATE:
                if (oppsCountLeft > 0 or oppsCountRight > 0) and oppsCountMiddle == 0:
                    return 1.
                else:
                    return 0
            elif cur_action == Action.BRAKE:
                if (oppsCountLeft > 0 or oppsCountRight > 0) and oppsCountMiddle == 0:
                    return -1.
                else:
                    return 0
            else:
                return 0

        value = getValue()

        print "SurpassNearbyOpponents {}".format(value)

        return super(SurpassNearbyOpponents, self).getFeatureValue(cur_action, value=value)
