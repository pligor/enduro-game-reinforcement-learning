import numpy as np
from enduro.action import Action
from road_category import RoadCategory
from extreme_position import ExtremePosition
from how_many_opponents import HowManyOpponents

class Sense(object):
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

        if pos == 1:
            return ExtremePosition().far_left
        elif pos == (self.gridWidth - 2):
            return ExtremePosition().far_right
        else:
            return ExtremePosition().elsewhere

    def getRoadCateg(self, prevGrid, action, newGrid):
        # print Action.toString(action)
        if self.isRoadTurningLeft(prevGrid, action, newGrid):
            return RoadCategory().turn_left
        elif self.isRoadTurningRight(prevGrid, action, newGrid):
            return RoadCategory().turn_right
        else:
            return RoadCategory().straight_ahead

    @staticmethod
    def isRoadTurningRight(prevGrid, action, newGrid):
        prevPos = np.argwhere(prevGrid[0] == 2)
        newPos = np.argwhere(newGrid[0] == 2)
        assert prevPos.shape == newPos.shape
        for i in range(len(prevPos.shape)):
            prevPos = prevPos[0]
            newPos = newPos[0]

        return (prevPos == newPos and action == Action.RIGHT) or (newPos < prevPos and action != Action.LEFT)

    # @staticmethod
    # def isRoadTurningRight(prevGrid, action, newGrid):
    #     prevPos = np.argwhere(prevGrid[0] == 2)
    #     newPos = np.argwhere(newGrid[0] == 2)
    #     assert prevPos.shape == newPos.shape
    #     for i in range(len(prevPos.shape)):
    #         prevPos = prevPos[0]
    #         newPos = newPos[0]
    #
    #     return (prevPos > newPos and action != Action.LEFT)

    @staticmethod
    def isRoadTurningLeft(prevGrid, action, newGrid):
        prevPos = np.argwhere(prevGrid[0] == 2)
        newPos = np.argwhere(newGrid[0] == 2)
        assert prevPos.shape == newPos.shape
        for i in range(len(prevPos.shape)):
            prevPos = prevPos[0]
            newPos = newPos[0]

        # return (prevPos < newPos and action != Action.LEFT) or (prevPos == newPos and action == Action.LEFT)
        return (prevPos < newPos and action != Action.RIGHT)

    def isOpponentInFront(self, grid, shift=0):
        ourCarPos = np.argwhere(grid[0] == 2)

        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        opponentPositions = np.argwhere(grid[1] == 1)

        camPos = np.min((np.max((0, ourCarPos + shift)), self.gridWidth - 1))

        return camPos in opponentPositions.flatten()

    def makeIndicesValid(self, inds):
        assert len(inds.shape) == 1
        length = len(inds)
        inds = np.max([inds, np.zeros(length)], axis=0)
        return np.min([inds, np.repeat(self.gridWidth - 1, length)], axis=0).astype(np.int)

    def isOpponentApproaching_old(self, cameraPos, prevGrid, newGrid):
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

    def isOpponentApproaching(self, cameraPos, prevGrid, newGrid):
        """camera position is where we place the camera (which column) to look for an approaching opponent from there
        camera is always in front of our car, the first line (not the zero-th)"""

        isOpponentFoundVeryFarAtPrevGrid = self.checkOpponentsInFront(prevGrid, (3, cameraPos), breadth=7)
        isOpponentFoundFarAtPrevGrid = self.checkOpponentsInFront(prevGrid, (2, cameraPos), breadth=5)

        isOpponentFoundNearAtNewGrid = self.checkOpponentsInFront(newGrid, (1, cameraPos), breadth=3)
        # isOpponentFoundFarAtNewGrid = self.checkOpponentsInFront(newGrid, (2, cameraPos), breadth=5)

        # print (isOpponentFoundVeryFarAtPrevGrid, isOpponentFoundVeryFarAtPrevGrid, isOpponentFoundNearAtNewGrid)

        return (isOpponentFoundVeryFarAtPrevGrid and isOpponentFoundNearAtNewGrid) or (
            isOpponentFoundFarAtPrevGrid and isOpponentFoundNearAtNewGrid
        )

    def oneCarInFrontApproaching(self, prevGrid, newGrid):
        ourCarPos = np.argwhere(newGrid[0] == 2)

        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        # return self.isOpponentInFront(newGrid) and self.isOpponentApproaching(ourCarPos, prevGrid, newGrid)
        return self.isOpponentApproaching(ourCarPos, prevGrid, newGrid)

    def oneCarInFrontLeftApproaching(self, prevGrid, newGrid):
        ourCarPos = np.argwhere(newGrid[0] == 2)
        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        camPos = np.max((0, np.min((self.gridWidth - 1, ourCarPos - 1))))

        # return self.isOpponentInFront(newGrid, shift=-1) and self.isOpponentApproaching(camPos, prevGrid, newGrid)
        return self.isOpponentApproaching(camPos, prevGrid, newGrid)

    def oneCarInFrontRightApproaching(self, prevGrid, newGrid):
        ourCarPos = np.argwhere(newGrid[0] == 2)
        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        camPos = np.max((0, np.min((self.gridWidth - 1, ourCarPos + 1))))

        # return self.isOpponentInFront(newGrid, shift=1) and self.isOpponentApproaching(camPos, prevGrid, newGrid)
        return self.isOpponentApproaching(camPos, prevGrid, newGrid)

    @staticmethod
    def getOurCarPos(grid):
        ourCarPos = np.argwhere(grid[0] == 2)

        for i in range(len(ourCarPos.shape)):
            ourCarPos = ourCarPos[0]

        return ourCarPos

    def isOpponentAtImmediate(self, grid, right_boolean):
        ourCarPos = self.getOurCarPos(grid)

        interestingPos = lambda shift: np.max((0, np.min((self.gridWidth - 1,
                                                          ourCarPos + (shift if right_boolean else -shift)
                                                          ))))

        return grid[0, interestingPos(1)] == 1 or grid[0, interestingPos(2)] == 1

    def checkOpponentsInFront(self, grid, coords, breadth=3):
        """coords are (line, column)"""
        assert breadth % 2 == 1
        line, column = coords
        assert line >= 0 and line < self.gridLength
        assert column >= 0 and column < self.gridWidth

        limit = int(np.floor(breadth / 2))

        inds = np.unique(self.makeIndicesValid(
            np.arange(-limit, limit + 1) + column
        ))

        return np.any(grid[line + 1, inds] == 1)

    def doesOpponentSurpasses(self, prevGrid, newGrid, curLine=0):
        # opponent surpasses from line 0 to next line
        # in prev grid line 0 is full and line 1 is empty and line 2 is empty
        # and
        # in next grid, line 0 is empty and (line 1 is full or line 2 is full)

        lineOpponents = np.argwhere(prevGrid[curLine] == 1).flatten()
        # print lineOpponents

        emptyInFront = []
        for i in lineOpponents:
            emptyInFront.append(
                (not self.checkOpponentsInFront(prevGrid, (curLine, i))) and
                (not self.checkOpponentsInFront(prevGrid, (curLine + 1, i)))
            )

        emptyInFront = np.array(emptyInFront)

        # print emptyInFront

        noOpponentsWherePreviouslyWere = np.all(newGrid[curLine, lineOpponents] != 1)

        fullInFront = []
        for i in lineOpponents:
            fullInFront.append(
                self.checkOpponentsInFront(newGrid, (curLine, i)) or self.checkOpponentsInFront(newGrid,
                                                                                                  (curLine + 1, i))
            )

        fullInFront = np.array(fullInFront)

        # print fullInFront
        assert len(fullInFront) == len(emptyInFront)
        fullAndEmptyChecks = False if len(fullInFront) == 0 else np.any(fullInFront & emptyInFront)

        # print type(noOpponentsWherePreviouslyWere)
        # print type(np.any(fullInFront & emptyInFront))

        return fullAndEmptyChecks and noOpponentsWherePreviouslyWere

    def countOpponents(self, grid, left_boolean):
        assert self.gridWidth % 2 == 0
        targetArea = grid[1:, :self.gridWidth/2] if left_boolean else grid[1:, self.gridWidth/2:]
        count = np.sum(targetArea == 1)
        howManyOpponents = HowManyOpponents()
        if count > howManyOpponents.maxcount:
            return howManyOpponents.many
        else:
            return str(count)

    def countOppsVarLen(self, grid, left_boolean, howFar, startFrom = 0):
        assert self.gridWidth % 2 == 0
        howFar = int(howFar)
        assert 0 <= howFar < self.gridLength
        startFrom = int(startFrom)
        assert 0 <= startFrom <= howFar

        targetArea = grid[startFrom:(howFar+1), :self.gridWidth/2] if left_boolean else \
            grid[startFrom:(howFar+1), self.gridWidth/2:]

        return np.sum(targetArea == 1)

if __name__ == "__main__":
    seed = 16011984
    rng = np.random  # .RandomState(seed=seed)
    # sense = Sense(rng=rng)
    sense = Sense(rng=rng)


    def testDoesOpponentSurpasses():
        prevGrid = sense.generateEmptyGrid()
        newGrid = prevGrid.copy()
        ourCarPos = sense.getOurCarPos(prevGrid)
        opponentPos = np.max((0, ourCarPos - 1))
        prevGrid[0, opponentPos] = 2 if opponentPos == ourCarPos else 1

        print prevGrid
        print
        print newGrid

        sense.doesOpponentSurpasses(prevGrid, newGrid)


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

    def testCountOpponents():
        grid = sense.generateEmptyGrid()
        print sense.countOpponents(grid, left_boolean=True)

        grid[5, 3] = 1
        print grid
        print sense.countOpponents(grid, left_boolean=True)

        grid[5, 2] = 1
        print grid
        print sense.countOpponents(grid, left_boolean=True)

        grid[4, 1] = 1
        print grid
        print sense.countOpponents(grid, left_boolean=True)

        grid[3, 0] = 1
        print grid
        print sense.countOpponents(grid, left_boolean=True)

        print sense.countOpponents(grid, left_boolean=False)

        grid[3, 9] = 1
        grid[3, 8] = 1
        grid[3, 7] = 1
        print grid
        print sense.countOpponents(grid, left_boolean=False)
