from __future__ import division

import numpy as np
from enduro.action import Action
from road_category import RoadCategory
from extreme_position import ExtremePosition
from how_many_opponents import HowManyOpponents
from sense import Sense
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

class Sensor(Sense):
    def __init__(self, rng):
        super(Sensor, self).__init__(rng)
        self.roadLength = self.gridLength + 1
        self.roadWidth = self.gridWidth + 1

    #def opponentsEffect

    def howMuchRoadTurning(self, road, action, pixelsThreshold = 10, factors = (0.01, 0.1, 0.2, 0.3, 1)):
        # road is turning left by some degree, or right. If road is turning left, left action should be
        # preferred. The rest in that order: accelerate, noop, brake, right
        # the same for if road is turning right
        middleCoord = self.roadWidth // 2

        roadReference = road[0, middleCoord]
        cameraReference = road[-1, middleCoord]
        horizonReference = np.array((cameraReference[0], roadReference[1]))

        roadNormalized = np.subtract(roadReference, cameraReference)
        horizonNormalized = np.subtract(horizonReference, cameraReference)

        cosSim = cosine_similarity(roadNormalized, horizonNormalized).flatten()[0]
        assert cosSim >= 0
        # large absolute value road is straight, small absolute value road is curved a lot

        roadRef_x = roadReference[0]
        camRef_x = cameraReference[0]

        if abs(roadRef_x - camRef_x) < pixelsThreshold:
            #straight ahead
            return 1
        else:
            assert roadRef_x != camRef_x
            isRoadTurningLeft = roadRef_x < camRef_x #is road turning left

            actionPreferred = Action.LEFT if isRoadTurningLeft else Action.RIGHT
            actionNotPreferred = Action.RIGHT if isRoadTurningLeft else Action.LEFT

            #print cosSim
            factors = (1. / cosSim) * np.array(factors)
            #print factors.shape

            if action == actionPreferred:
                return factors[4]
            elif action == Action.ACCELERATE:
                return factors[3]
            elif action == Action.NOOP:
                return factors[2]
            elif action == Action.BRAKE:
                return factors[1]
            elif action == actionNotPreferred:
                return factors[0]

        raise AssertionError  #invalid action detected

    def opponentsBeside(self, grid, action, factors = (0.01, 0.1, 0.2, 0.5, 0.9)):
        # detect if opponents are found on the left and the action is left then this should be a small value
        # if opponents are found on the right and the action is right then small value
        # brake should have a smaller impact, noop smaller
        # accelerate is preferred, the opposite action (right, here) might be also preferred
        opponentsOnTheLeft = self.isOpponentAtImmediate(grid, right_boolean=False)
        opponentsOnTheRight = self.isOpponentAtImmediate(grid, right_boolean=True)

        if opponentsOnTheLeft and (not opponentsOnTheRight):
            if action == Action.LEFT:
                return factors[0]
            elif action == Action.BRAKE:
                return factors[1]
            elif action == Action.NOOP:
                return factors[2]
            elif action == Action.RIGHT:
                return factors[4]
            elif action == Action.ACCELERATE:
                return factors[3]

        elif (not opponentsOnTheLeft) and opponentsOnTheRight:
            if action == Action.RIGHT:
                return factors[0]
            elif action == Action.BRAKE:
                return factors[1]
            elif action == Action.NOOP:
                return factors[2]
            elif action == Action.LEFT:
                return factors[4]
            elif action == Action.ACCELERATE:
                return factors[3]

        elif opponentsOnTheLeft and opponentsOnTheRight:
            if action == Action.ACCELERATE:
                return factors[4]
            elif action == Action.LEFT or action == Action.RIGHT:
                return factors[0]
            elif action == Action.NOOP:
                return factors[1]
            elif action == Action.BRAKE:
                return factors[2]

        else:
            return 0  #do not let it play a role

        raise AssertionError  #an unexpected action was used

    def distanceFromCentre(self, grid, action, factor = 2.):
        # being in the centre [4 or 5 position] we need the highest value, the lowest value at the edges
        # 4.5 - 0 = 4.5, 4.5 - 9 = -4.5, while 4.5 - 4 = 0.5 and 4.5 - 5 = -0.5
        # if negative means we are on the right, action Left should bring it a larger value, action Right a smaller
        # positive means we are on the left, action Right larger value, action Left smaller
        # action accelerate or break or noop are neutral
        carPos = self.getOurCarPos(grid)
        middlePos = (self.gridWidth - 1) / 2.
        distance = middlePos - carPos
        areWeOnTheRight =  distance < 0
        areWeOnTheLeft = not areWeOnTheRight

        distance **= 2

        if (areWeOnTheRight and action == Action.RIGHT) or ( areWeOnTheLeft and action == Action.LEFT):
            return distance / factor
        elif (areWeOnTheRight and action == Action.LEFT) or (areWeOnTheLeft and action == Action.RIGHT):
            return distance * factor
        else:
            return distance

    ####################################################################################################################

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
