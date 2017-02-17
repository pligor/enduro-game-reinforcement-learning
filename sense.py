import numpy as np
from enduro.action import Action
from road_category import RoadCategory
from extreme_position import ExtremePosition


class Sense:
    def __init__(self, rng):
        self.rng = rng
        self.gridLength = 11
        self.gridWidth = 10

    def generateGridWithOneCarInFront(self):
        sampleGrid = np.zeros((self.gridLength, self.gridWidth))
        mycarInd = self.rng.randint(0, sampleGrid.shape[1])
        sampleGrid[0, mycarInd] = 2
        opponentRandInd = \
            np.random.permutation([np.max((mycarInd - 1, 0)), mycarInd, np.min((mycarInd + 1, self.gridWidth - 1))])[0]
        sampleGrid[1, opponentRandInd] = 1
        return sampleGrid

    def generateEmptyGrid(self):
        sampleGrid = np.zeros((self.gridLength, self.gridWidth))
        mycarInd = self.rng.randint(0, sampleGrid.shape[1])
        sampleGrid[0, mycarInd] = 2
        return sampleGrid

    def getExtremePosition(self, latestGrid):
        pos = np.argwhere(latestGrid[0] == 2)

        for i in range(len(pos.shape)):
            pos = pos[0]

        if pos == 0:
            return ExtremePosition().far_right
        elif pos == (self.gridWidth - 1):
            return ExtremePosition().far_left
        else:
            return ExtremePosition().elsewhere

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

    def isOpponentInFront(self, grid, shift = 0):
        ourCarPos = np.argwhere(grid[0] == 2)

        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        opponentPositions = np.argwhere(grid[1] == 1)

        camPos = np.min( (np.max((0, ourCarPos + shift)), self.gridWidth -1) )

        return camPos in opponentPositions.flatten()

    def makeIndicesValid(self, inds):
        assert len(inds.shape) == 1
        length = len(inds)
        inds = np.max([inds, np.zeros(length)], axis=0)
        return np.min([inds, np.repeat(self.gridWidth - 1, length)], axis=0).astype(np.int)

    def isOpponentApproaching(self, cameraPos, prevGrid, newGrid):
        """camera position is where we place the camera (which column) to look for an approaching opponent from there
        camera is always in front of our car, the first line (not the zero-th)"""
        if newGrid[1, cameraPos] == 1:
            oneStepBackInds = np.array(
                [np.max((0, cameraPos - 1)), cameraPos, np.min((cameraPos + 1, self.gridWidth - 1))])

            oneStepBackCriteria = np.any(prevGrid[2, oneStepBackInds] == 1)

            twoStepBackInds = self.makeIndicesValid(
                np.array([cameraPos - 2, cameraPos - 1, cameraPos, cameraPos + 1, cameraPos + 2])
            )

            twoStepBackCriteria = np.any(prevGrid[3, twoStepBackInds] == 1)

            return oneStepBackCriteria or twoStepBackCriteria
        else:
            return False

    def oneCarInFrontApproaching(self, prevGrid, newGrid):
        ourCarPos = np.argwhere(newGrid[0] == 2)

        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        return self.isOpponentInFront(newGrid) and self.isOpponentApproaching(ourCarPos, prevGrid, newGrid)

    def oneCarInFrontRightApproaching(self, prevGrid, newGrid):
        ourCarPos = np.argwhere(newGrid[0] == 2)

        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        camPos = np.max( (0, np.min( (self.gridWidth - 1, ourCarPos - 1) ) )  )

        return self.isOpponentInFront(newGrid, shift=-1) and self.isOpponentApproaching(camPos, prevGrid, newGrid)

    def oneCarInFrontLeftApproaching(self, prevGrid, newGrid):
        ourCarPos = np.argwhere(newGrid[0] == 2)

        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        camPos = np.max((0, np.min((self.gridWidth - 1, ourCarPos + 1))))

        return self.isOpponentInFront(newGrid, shift=1) and self.isOpponentApproaching(camPos, prevGrid, newGrid)

    @staticmethod
    def getOurCarPos(grid):
        ourCarPos = np.argwhere(grid[0] == 2)

        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        return ourCarPos

    def isOpponentAtImmediate(self, grid, left):
        ourCarPos = self.getOurCarPos(grid)

        interestingPos = ourCarPos + (1 if left else -1)
        interestingPos = np.max( (0, np.min( (self.gridWidth - 1,  interestingPos) ) ) )

        return grid[0, interestingPos] == 1


if __name__ == "__main__":
    seed = 16011984
    rng = np.random  # .RandomState(seed=seed)
    # sense = Sense(rng=rng)
    sense = Sense(rng=rng)

    def testIsOpponentApproaching():
        prevGrid = sense.generateEmptyGrid()
        newGrid = sense.generateEmptyGrid()
        prevGrid[3, 9] = 1
        # prevGrid[2, 9] = 1
        newGrid[1, 5] = 1

        print prevGrid
        print sense.isOpponentApproaching(5, prevGrid, newGrid)
        print newGrid


    def testMakeIndicesValid():
        print sense.makeIndicesValid(np.array([-2, 4, 10, 100, 5, 8, -7]))


    def testIsOpponentInFront():
        grid = sense.generateGridWithOneCarInFront()
        print grid
        print sense.isOpponentInFront(grid)


    def testExtremePosition():
        grid = sense.generateEmptyGrid()
        print grid
        print sense.getExtremePosition(grid)


    def testRoadCateg():
        actionSet = [Action.ACCELERATE, Action.BREAK, Action.LEFT, Action.RIGHT]
        prevGrid = sense.generateEmptyGrid()
        newGrid = sense.generateEmptyGrid()
        randomAct = actionSet[rng.randint(0, len(actionSet))]
        print prevGrid
        print sense.getRoadCateg(prevGrid, randomAct, newGrid)
        print Action.toString(randomAct)
        print newGrid


    def testTurningRight():
        prevGrid = sense.generateEmptyGrid()
        newGrid = sense.generateEmptyGrid()
        print prevGrid
        print sense.isRoadTurningRight(prevGrid, Action.BREAK, newGrid)
        print newGrid


    def testTurningLeft():
        prevGrid = sense.generateEmptyGrid()
        newGrid = sense.generateEmptyGrid()
        print prevGrid
        print sense.isRoadTurningLeft(prevGrid, Action.BREAK, newGrid)
        print newGrid


    testIsOpponentInFront()