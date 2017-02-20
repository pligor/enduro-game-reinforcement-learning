import numpy as np
from enduro_data_types import tuple_dt
from agent_with_senses import AgentWithSenses

class AgentWithShortOrizonSenses(AgentWithSenses):
    def __init__(self, rng):
        super(AgentWithShortOrizonSenses, self).__init__()
        # Add member variables to your class here
        self.states = np.load('enduro_states_short_horizon.npy')

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
