import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
from sense import Sense
import numpy as np
from enduro_data_types import tuple_dt
from agent_with_short_orizon_senses import AgentWithShortOrizonSenses

class AgentWithLongOrizonSenses(AgentWithShortOrizonSenses):
    def __init__(self, rng):
        super(AgentWithLongOrizonSenses, self).__init__(rng)
        # Add member variables to your class here
        self.sensor = Sense(rng)
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

        allSenses = np.array((roadCateg,
                              extremePos,
                              isCarInFrontApproaching,
                              isCarInFrontRightApproaching,
                              isCarInFrontLeftApproaching,
                              areOpponentsSurpassing,
                              isOpponentAtImmediateLeft,
                              isOpponentAtImmediateRight), dtype=tuple_dt)

        #print allSenses

        return allSenses

    def getStateById(self, stateId):
        ss = [s for s in self.states if s['id'] == stateId]
        assert len(ss) == 1
        return ss[0]
