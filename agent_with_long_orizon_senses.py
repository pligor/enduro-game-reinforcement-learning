import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
from long_sense import LongSense
import numpy as np
from enduro_data_types import long_tuple_dt
from agent_with_short_orizon_senses import AgentWithShortOrizonSenses

class AgentWithLongOrizonSenses(AgentWithShortOrizonSenses):
    def __init__(self, rng):
        super(AgentWithLongOrizonSenses, self).__init__(rng)
        # Add member variables to your class here
        self.sensor = LongSense(rng)
        self.states = np.load('enduro_states_long_horizon.npy')

    def getAllSenses(self, prevGrid, action, newGrid):
        roadCateg = self.sensor.getRoadCateg(prevGrid, action, newGrid)
        extremePos = self.sensor.getExtremePosition(latestGrid=newGrid)
        isCarInFrontApproaching = self.sensor.oneCarInFrontApproaching(prevGrid, newGrid)
        isCarInFrontRightApproaching = self.sensor.oneCarInFrontRightApproaching(prevGrid, newGrid)
        isCarInFrontLeftApproaching = self.sensor.oneCarInFrontLeftApproaching(prevGrid, newGrid)
        areOpponentsSurpassing = self.sensor.doesOpponentSurpasses(prevGrid, newGrid)
        isOpponentAtImmediateLeft = self.sensor.isOpponentAtImmediate(newGrid, right_boolean=False)
        isOpponentAtImmediateRight = self.sensor.isOpponentAtImmediate(newGrid, right_boolean=True)
        countOpponentsRight = self.sensor.countOpponents(newGrid, left_boolean=False)
        countOpponentsLeft = self.sensor.countOpponents(newGrid, left_boolean=True)

        allSenses = np.array((roadCateg,
                              extremePos,
                              isCarInFrontApproaching,
                              isCarInFrontRightApproaching,
                              isCarInFrontLeftApproaching,
                              areOpponentsSurpassing,
                              isOpponentAtImmediateLeft,
                              isOpponentAtImmediateRight,
                              countOpponentsRight,
                              countOpponentsLeft), dtype=long_tuple_dt)

        return allSenses
