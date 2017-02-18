import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
from sense import Sense
import numpy as np
from enduro_data_types import tuple_dt

class AgentWithShortOrizonSenses(Agent):
    def __init__(self, rng):
        super(AgentWithShortOrizonSenses, self).__init__()
        # Add member variables to your class here

        self.rng = rng

        self.sensor = Sense(rng)

        self.states = np.load('enduro_states_short_horizon.npy')

    def getStateIds(self):
        return [s['id'] for s in self.states]

    def getStateIdBySensing(self, prevGrid, action, newGrid):
        allSenses = self.getAllSenses(prevGrid, action, newGrid)
        # a = allSenses.copy()
        # a['road'] = "333"
        # print a == allSenses

        stateIds = [s['id'] for s in self.states if s['tuple'] == allSenses]

        assert len(stateIds) == 1
        stateId = stateIds[0]

        #print stateId
        return stateId

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
