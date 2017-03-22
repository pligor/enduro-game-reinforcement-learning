from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature
from sensor import Sensor
from scipy.stats import multivariate_normal as mvnorm
from collections import OrderedDict


class GaussianThreatFeature(Feature):  # ContrainedFeature
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = self.getPriorsPerAction()

        self.sensor = Sensor(rng=rng)

        cov_magnitude = 4  # small more peaky, large more take into account the neighborhood
        self.offset = 1
        self.row_max_threat = 2
        self.FRONT_ROW = 1  # this is capital because it is a constant really

        self.cov = np.array([[cov_magnitude, 0], [0, cov_magnitude]])  # no correlation between axes

        super(GaussianThreatFeature, self).__init__()  # rng=rng

    def getPriorsPerAction(self):
        return OrderedDict([
            (Action.ACCELERATE, -10.),
            (Action.RIGHT, 20.),
            (Action.LEFT, 20.),
            (Action.BRAKE, 10.),
            (Action.NOOP, 5.),
        ])

    def __getPDFgaussian(self, carPos, oppCoords):
        mean = np.array([self.row_max_threat, carPos])

        return mvnorm.pdf(oppCoords, mean=mean, cov=self.cov)

    def get_opps_coords(self, grid, carPos):
        return self.sensor.getOpponentsCoords(grid=grid)

    def getFeatureValue(self, cur_action, **kwargs):
        grid = kwargs['grid']
        carPos = self.sensor.getOurCarPos(grid)
        opponents_coords = self.get_opps_coords(grid=grid, carPos=carPos)

        total_threat = 0

        for opp_coords in opponents_coords:
            if opp_coords[0] >= self.FRONT_ROW:  # considering ONLY opponents from row 1 and beyond
                total_threat += self.__getPDFgaussian(carPos=carPos, oppCoords=opp_coords)

        total_value = -total_threat
        q_value = super(GaussianThreatFeature, self).getFeatureValue(cur_action, value=total_value)

        # print "q value {}".format(q_value)

        return q_value + self.offset


class GaussianThreatLeftFeature(GaussianThreatFeature):  # ContrainedFeature
    def __init__(self, corresponding_action, rng):
        super(GaussianThreatLeftFeature, self).__init__(
            corresponding_action=corresponding_action, rng=rng)

    def get_opps_coords(self, grid, carPos):
        opps_coords = self.sensor.getOpponentsCoords(grid=grid)
        return opps_coords[np.argwhere(opps_coords[:, 1] <= carPos).flatten()]

    def getPriorsPerAction(self):
        return OrderedDict([
            (Action.ACCELERATE, -10.),
            (Action.RIGHT, 20.),
            (Action.LEFT, -20.),
            (Action.BRAKE, 10.),
            (Action.NOOP, 5.),
        ])


class GaussianThreatRightFeature(GaussianThreatFeature):  # ContrainedFeature
    def __init__(self, corresponding_action, rng):
        super(GaussianThreatRightFeature, self).__init__(
            corresponding_action=corresponding_action, rng=rng)

    def get_opps_coords(self, grid, carPos):
        opps_coords = self.sensor.getOpponentsCoords(grid=grid)
        return opps_coords[np.argwhere(opps_coords[:, 1] >= carPos).flatten()]

    def getPriorsPerAction(self):
        return OrderedDict([
            (Action.ACCELERATE, -10.),
            (Action.RIGHT, -20.),
            (Action.LEFT, 20.),
            (Action.BRAKE, 10.),
            (Action.NOOP, 5.),
        ])