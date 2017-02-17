import numpy as np
from enduro.action import Action
from road_category import RoadCategory


class Sense:
    def generateGrid(self):
        sampleGrid = np.zeros((11, 10))
        mycarInd = self.rng.randint(0, sampleGrid.shape[1])
        sampleGrid[0, mycarInd] = 2
        return sampleGrid

    def __init__(self, rng):
        self.rng = rng



    def getRoadCateg(self, prevGrid, action, newGrid):
        if self.isRoadTurningLeft(prevGrid, action, newGrid):
            return RoadCategory().turn_left
        elif self.isRoadTurningRight(prevGrid, action, newGrid):
            return RoadCategory().turn_right
        else:
            return RoadCategory.straight_ahead

    @staticmethod
    def isRoadTurningRight(prevGrid, action, newGrid):
        prevPos = np.argwhere(prevGrid[0] == 2)
        newPos = np.argwhere(newGrid[0] == 2)

        assert prevPos.shape == newPos.shape

        for i in range(len(prevPos.shape)):
            prevPos = prevPos[0]
            newPos = newPos[0]

        print prevPos, newPos
        return prevPos < newPos and action != Action.LEFT

    @staticmethod
    def isRoadTurningLeft(prevGrid, action, newGrid):
        prevPos = np.argwhere(prevGrid[0] == 2)
        newPos = np.argwhere(newGrid[0] == 2)

        assert prevPos.shape == newPos.shape

        for i in range(len(prevPos.shape)):
            prevPos = prevPos[0]
            newPos = newPos[0]

        print prevPos, newPos
        return prevPos > newPos and action != Action.RIGHT


if __name__ == "__main__":
    seed = 16011984
    rng = np.random  # .RandomState(seed=seed)
    # sense = Sense(rng=rng)
    sense = Sense(rng=rng)

    actionSet = [Action.ACCELERATE, Action.BREAK, Action.LEFT, Action.RIGHT]

    def testRoadCateg():
        prevGrid = sense.generateGrid()
        newGrid = sense.generateGrid()
        randomAct = actionSet[rng.randint(0, len(actionSet))]
        print prevGrid
        print sense.getRoadCateg(prevGrid, randomAct, newGrid)
        print Action.toString(randomAct)
        print newGrid


    # print sense.generateGrid()
    def testTurningRight():
        prevGrid = sense.generateGrid()
        newGrid = sense.generateGrid()
        print prevGrid
        print sense.isRoadTurningRight(prevGrid, Action.BREAK, newGrid)
        print newGrid


    def testTurningLeft():
        prevGrid = sense.generateGrid()
        newGrid = sense.generateGrid()
        print prevGrid
        print sense.isRoadTurningLeft(prevGrid, Action.BREAK, newGrid)
        print newGrid
