from __future__ import division
import numpy as np
from enduro.action import Action
from feature_base import Feature, ContrainedFeature
from sensor import Sensor
from scipy.stats import multivariate_normal as mvnorm

class GaussianThreatFeature(Feature):  #ContrainedFeature
    def __init__(self, corresponding_action, rng):
        self.corresponding_action = corresponding_action

        self.priors_per_action = {
            Action.ACCELERATE: -10,
            Action.RIGHT: 20,
            Action.LEFT: 20,
            Action.BRAKE: 10,
            Action.NOOP: 5,
        }

        self.sensor = Sensor(rng=rng)

        cov_magnitude = 3  # small more peaky, large more take into account the neighborhood

        self.row_max_threat = 2
        self.FRONT_ROW = 1  #this is capital because it is a constant really

        self.cov = np.array([[cov_magnitude, 0], [0, cov_magnitude]])  #no correlation between axes

        super(GaussianThreatFeature, self).__init__()  #rng=rng

    def __getPDFgaussian(self, carPos, oppCoords):
        mean = np.array([self.row_max_threat, carPos])

        return mvnorm.pdf(oppCoords, mean=mean, cov=self.cov)

    def getFeatureValue(self, cur_action, **kwargs):
        grid = kwargs['grid']
        opponents_coords = self.sensor.getOpponentsCoords(grid=grid)
        carPos = self.sensor.getOurCarPos(grid)

        total_threat = 0

        for opp_coords in opponents_coords:
            if opp_coords[0] >= self.FRONT_ROW: # considering ONLY opponents from row 1 and beyond
                total_threat += self.__getPDFgaussian(carPos=carPos, oppCoords=opp_coords)

        total_value = -total_threat

        return super(GaussianThreatFeature, self).getFeatureValue(cur_action, value=total_value)