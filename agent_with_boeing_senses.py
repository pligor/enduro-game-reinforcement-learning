import numpy as np
from enduro_data_types import tuple_boeing_dt
from agent_with_senses import AgentWithSenses
from how_many_opponents_discrete import HowManyOpponentsDiscrete

class AgentWithBoeingSenses(AgentWithSenses):
    def __init__(self, rng):
        self.states = np.load('enduro_boeing_states.npy')

        super(AgentWithBoeingSenses, self).__init__(rng)

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
        roadCateg = self.sensor.getRoadCateg(prevGrid, action, newGrid)
        extremePos = self.sensor.getExtremePosition(latestGrid=newGrid)

        oppNearLeft = self.__countToTextClass(
            self.sensor.countOppsVarLen(newGrid, left_boolean=True, howFar=5, startFrom=0))
        oppNearRight = self.__countToTextClass(
            self.sensor.countOppsVarLen(newGrid, left_boolean=False, howFar=5, startFrom=0))
        oppFarLeft = self.__countToTextClass(
            self.sensor.countOppsVarLen(newGrid, left_boolean=True, howFar=11, startFrom=6))
        oppFarRight = self.__countToTextClass(
            self.sensor.countOppsVarLen(newGrid, left_boolean=False, howFar=11, startFrom=6))

        areOpponentsSurpassing = self.sensor.doesOpponentSurpasses(prevGrid, newGrid)
        isOpponentAtImmediateLeft = self.sensor.isOpponentAtImmediate(newGrid, right_boolean=False)
        isOpponentAtImmediateRight = self.sensor.isOpponentAtImmediate(newGrid, right_boolean=True)

        allSenses = np.array((roadCateg,
                              extremePos,
                              oppNearLeft,
                              oppNearRight,
                              oppFarLeft,
                              oppFarRight,
                              areOpponentsSurpassing,
                              isOpponentAtImmediateLeft,
                              isOpponentAtImmediateRight), dtype=tuple_boeing_dt)

        #print allSenses

        return allSenses
