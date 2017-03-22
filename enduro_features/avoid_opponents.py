from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature, PlainFeature
from sensor import Sensor
from collections import OrderedDict


class AvoidOppsFarAway(ContrainedFeature, PlainFeature):
    def __init__(self, rng):
        self.prior_weight = 5.

        self.sensor = Sensor(rng=rng)

        super(AvoidOppsFarAway, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        grid = kwargs['grid']

        carPos = self.sensor.getOurCarPos(grid=grid)
        oppsFarLeft = self.sensor.countOppsInRelationToCar(grid=grid, carPos=carPos,
                                                           startRow=6, howFar=10, shift=-2)
        oppsFarRight = self.sensor.countOppsInRelationToCar(grid=grid, carPos=carPos,
                                                            startRow=6, howFar=10, shift=+2)

        # if action is correct the largest the difference the most correct the action
        diff = (oppsFarLeft - oppsFarRight) ** 2

        def getValue():
            if oppsFarLeft > 0 or oppsFarRight > 0:
                if cur_action == Action.ACCELERATE:
                    return -1.
                elif cur_action == Action.BRAKE:
                    return 1.
                elif cur_action == Action.RIGHT:
                    if oppsFarLeft > oppsFarRight:
                        return diff
                    elif oppsFarLeft < oppsFarRight:
                        return -diff
                    else:
                        return 0
                elif cur_action == Action.LEFT:
                    if oppsFarLeft < oppsFarRight:
                        return diff
                    elif oppsFarLeft > oppsFarRight:
                        return -diff
                    else:
                        return 0
                else:
                    return 0
            else:
                if cur_action == Action.ACCELERATE:
                    return 1.
                elif cur_action == Action.BRAKE:
                    return -1.
                else:
                    return 0

        value = getValue()

        print "AvoidOppsFarAway {}".format(value)

        return super(AvoidOppsFarAway, self).getFeatureValue(cur_action, value=value)


class AvoidOppsNearby(ContrainedFeature, PlainFeature):
    def __init__(self, rng):
        self.prior_weight = 5.

        self.sensor = Sensor(rng=rng)

        super(AvoidOppsNearby, self).__init__()

    def getFeatureValue(self, cur_action, **kwargs):
        grid = kwargs['grid']

        carPos = self.sensor.getOurCarPos(grid=grid)
        oppsCountLeft = self.sensor.countOppsInRelationToCar(grid=grid, carPos=carPos, howFar=5, shift=-1)
        oppsCountRight = self.sensor.countOppsInRelationToCar(grid=grid, carPos=carPos, howFar=5, shift=+1)

        # if action is correct the largest the difference the most correct the action
        diff = (oppsCountLeft - oppsCountRight) ** 2

        def getValue():
            if (oppsCountLeft > oppsCountRight and cur_action == Action.RIGHT) or (
                            oppsCountLeft < oppsCountRight and cur_action == Action.LEFT):
                return diff
            elif (oppsCountLeft > oppsCountRight and cur_action == Action.LEFT) or (
                            oppsCountLeft < oppsCountRight and cur_action == Action.RIGHT):
                return -diff
            else:
                return 0

        value = getValue()

        # print "AvoidOppsNearby {}".format(value)

        return super(AvoidOppsNearby, self).getFeatureValue(cur_action, value=value)
