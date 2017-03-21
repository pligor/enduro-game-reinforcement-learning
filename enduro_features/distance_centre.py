from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature
from sensor import Sensor


class BeingInTheCentreIsBetter(ContrainedFeature, Feature):
    def __init__(self, rng):
        super(BeingInTheCentreIsBetter, self).__init__()
        self.sensor = Sensor(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        return super(BeingInTheCentreIsBetter, self).getFeatureValue(cur_action, value=kwargs['distance'])


class MoveRightWhenLeft(BeingInTheCentreIsBetter):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = {
            Action.ACCELERATE: 0.1,
            Action.RIGHT: 0.9,
            Action.LEFT: -0.8,
            Action.BRAKE: 0.1,
            Action.NOOP: 0.1,
        }

        super(MoveRightWhenLeft, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        distance, areWeOnTheRight = self.sensor.distanceFromCentre(grid=kwargs['grid'])

        return super(MoveRightWhenLeft, self).getFeatureValue(cur_action, distance=distance) \
            if not areWeOnTheRight else 0  # does not play role


class MoveLeftWhenRight(BeingInTheCentreIsBetter):
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = {
            Action.ACCELERATE: 0.1,
            Action.RIGHT: -0.8,
            Action.LEFT: 0.9,
            Action.BRAKE: 0.1,
            Action.NOOP: 0.1,
        }

        super(MoveLeftWhenRight, self).__init__(rng=rng)

    def getFeatureValue(self, cur_action, **kwargs):
        distance, areWeOnTheRight = self.sensor.distanceFromCentre(grid=kwargs['grid'])

        return super(MoveLeftWhenRight, self).getFeatureValue(cur_action, distance=distance) \
            if areWeOnTheRight else 0  # does not play role
