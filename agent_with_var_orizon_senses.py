import numpy as np
from enduro_data_types import tuple_v2_dt
from agent_with_senses import AgentWithSenses
from how_many_opponents_discrete import HowManyOpponentsDiscrete


class AgentWithVarOrizonSenses(AgentWithSenses):
    def __init__(self, rng, howFar):
        self.states = np.load('enduro_states_short_v2_horizon.npy')
        self.howFar = howFar
        super(AgentWithVarOrizonSenses, self).__init__(rng)

    @staticmethod
    def __countToTextClass(count):
        count = int(count)
        assert count >= 0
        if count == 0:
            return HowManyOpponentsDiscrete().none
        elif 1 <= count <= 2:
            return HowManyOpponentsDiscrete().afew
        else:
            return HowManyOpponentsDiscrete().alot

    def getAllSenses(self, prevGrid, action, newGrid):
        pos = self.sensor.getExtremePosition(latestGrid=newGrid)
        oppLeft = self.__countToTextClass(self.sensor.countOppsVarLen(newGrid, left_boolean=True, howFar=self.howFar))
        oppRight = self.__countToTextClass(self.sensor.countOppsVarLen(newGrid, left_boolean=False, howFar=self.howFar))
        areOpponentsSurpassing = self.sensor.doesOpponentSurpasses(prevGrid, newGrid)

        return np.array((pos,
                         oppLeft,
                         oppRight,
                         areOpponentsSurpassing), dtype=tuple_v2_dt)

