from __future__ import division

import numpy as np
from enduro.action import Action
from road_category import RoadCategory
from extreme_position import ExtremePosition
from how_many_opponents import HowManyOpponents
from sense import Sense
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sys import maxint
from itertools import permutations


class Sensor(Sense):
    def __init__(self, rng):
        super(Sensor, self).__init__(rng)
        self.roadLength = self.gridLength + 1
        self.roadWidth = self.gridWidth + 1

    @staticmethod
    def getOpponentsCoords(grid):
        return np.argwhere(grid == 1)

    @staticmethod
    def getCarsDist(car_a, car_b):
        return np.sum((car_a[:2] - car_b[:2]) ** 2)

    def getClosestCar(self, car, otherCars):
        assert len(otherCars) > 0
        curMinDist = maxint
        curClosestOtherCar = None
        curMinInd = None
        for curInd, otherCar in enumerate(otherCars):
            curDist = self.getCarsDist(otherCar, car)
            if curDist < curMinDist:
                curClosestOtherCar = otherCar
                curMinInd = curInd
            curMinDist = min(curDist, curMinDist)

        assert curClosestOtherCar is not None
        assert curMinInd is not None

        return curClosestOtherCar, curMinDist, curMinInd

    def getClosestCarForAllCars(self, cars, otherCars):
        curAssignment = []
        assert len(otherCars) > 0  # make sure that it will run at least once
        otherCars = list(otherCars)  # convert numpy array lines in items of a python list to be able and delete
        for car in cars:
            if len(otherCars) == 0:
                break
            curClosestOtherCar, curMinDist, curMinInd = self.getClosestCar(car, otherCars)
            curAssignment.append((car, curClosestOtherCar, curMinDist))
            del otherCars[curMinInd]

        return curAssignment

    def getPossibleAssignments(self, cars, otherCars):
        possibleAssignments = []

        for curCarsPermutation in permutations(cars):
            possibleAssignments.append(
                self.getClosestCarForAllCars(curCarsPermutation, otherCars=otherCars)
            )

        return possibleAssignments

    @staticmethod
    def getBestAssignment(assignments):
        """returns an array of tuples (car, othercar, distance)"""
        bestAssignment = None
        minTotalDist = maxint
        for assignment in assignments:
            curTotalDist = np.sum([aa[-1] for aa in assignment])
            if curTotalDist < minTotalDist:
                bestAssignment = assignment
            minTotalDist = min(minTotalDist, curTotalDist)

        assert bestAssignment is not None

        return bestAssignment

    @staticmethod
    def getMagnitudeDifferencesFromBestAssignment(bestAssignment):
        """positive means car is approaching, negative means car is moving away from us"""
        areaCouples = [(carCouple[0][2:4], carCouple[1][2:4]) for carCouple in bestAssignment]
        magnitudeCouples = [(xx[0][0] * xx[0][1], xx[1][0] * xx[1][1]) for xx in areaCouples]
        magnitudeDifferences = [xx[0] - xx[1] for xx in magnitudeCouples]
        return magnitudeDifferences

    def getRelativeSpeedToOpponent(self, prevCars, cars):
        prevOpps = np.array(prevCars['others'])
        opps = np.array(cars['others'])
        if len(opps) > 0 and len(prevOpps) > 0:
            # we want to find the car which is nearest to the previous car

            # we are taking the euclidean distance between one car of current list with all cars of previous list
            # we are doing this for all cars of current list
            # we want to minimize the overall
            bestAssignment = self.getBestAssignment(self.getPossibleAssignments(cars=opps, otherCars=prevOpps))

            magnitudeDiffs = self.getMagnitudeDifferencesFromBestAssignment(bestAssignment)
            #print "magnitudeDiffs: {}".format(magnitudeDiffs)

            meanMagnDiff = np.mean(magnitudeDiffs)
            print "meanMagnDiff: {}".format(meanMagnDiff)
        else:
            return None

    def getEstimatedSpeed(self, prevGrid, grid):
        """positive means opponents approaching, negative means they are moving away
        and values range from ~ -3 to ~ +5 approximately"""
        prevAveragePos = self.getAveragePositionOfFirstRowOpps(prevGrid)
        averagePos = self.getAveragePositionOfFirstRowOpps(grid)
        if averagePos is None or prevAveragePos is None:
            return None
        else:
            return - (averagePos - prevAveragePos)

    def getAveragePositionOfFirstRowOpps(self, grid):
        oppPos = []
        for col in grid[1:].T:
            opps = np.argwhere(col == self.opponent_symbol).flatten()
            if len(opps) > 0:
                oppPos.append(opps[0])

        if len(oppPos) == 0:
            return None
        else:
            return np.mean(oppPos)

    def senseOppsInFrontOfCar(self, grid, carPos, width=3, factor=1., noOpponentsFactor=3):
        startFrom = 1

        assert width % 2 == 1, "only odd (symmetric around the car) widths are accepted"
        extraCols = width // 2

        fromCol = max(carPos - extraCols, 0)
        toCol = min(carPos + extraCols + 1, self.gridWidth)

        targetArea = grid[startFrom:, fromCol:toCol]

        opps = np.argwhere(targetArea == 1)

        effects = np.sum([1. / (opp[0] + 1) ** factor for opp in opps])

        if effects == 0:
            maxObservation = len(targetArea) ** factor
            return maxObservation * noOpponentsFactor
        else:
            return 1. / effects

    def getAngleAndMagnitudeOfOpponentFromEnv(self, cars, road, opp_index):
        self_vec = cars['self'][:2]

        invalid = (None, None, None)

        if "others" in cars:
            opponents = cars['others']
            opponentsCount = len(opponents)

            horizon_vec = self.getHorizonVec(road)

            if opponentsCount > opp_index:
                cur_opp = opponents[opp_index]

                magnitude = cur_opp[2] * cur_opp[3]

                cos_sim, isOpponentLeft = self.getAngleOfOpponent(
                    opponent=cur_opp,
                    horizon_vec=horizon_vec, self_vec=self_vec
                )

                return cos_sim, isOpponentLeft, magnitude
            else:
                return invalid
        else:
            return invalid

    def getAngleOfOpponentFromEnv(self, cars, road, opp_index):
        self_vec = cars['self'][:2]

        invalid = (0, None)

        if "others" in cars:
            opponents = cars['others']
            opponentsCount = len(opponents)

            horizon_vec = self.getHorizonVec(road)

            if opponentsCount > opp_index:
                return self.getAngleOfOpponent(opponent=opponents[opp_index],
                                               horizon_vec=horizon_vec, self_vec=self_vec)
            else:
                return invalid
        else:
            return invalid

    def getAngleOfOpponent(self, opponent, self_vec, horizon_vec):
        opponent_vec = opponent[:2]
        opp_car_vec = np.subtract(self_vec, opponent_vec)

        hor_car_vec = np.subtract(self_vec, horizon_vec)

        # same_x_far_y = np.array((self_vec[0], 0))  # 0 is the maximum point in the screen
        # recall the y axis in the screen counts from top to bottom
        # self_perpendicular = np.subtract(self_vec, same_x_far_y)
        # axis = np.array([1, 0])
        # isOpponentLeft = cosine_similarity(axis.reshape(1, -1), opp_car_vec.reshape(1, -1)) > \
        #                  cosine_similarity(axis.reshape(1, -1), hor_car_vec.reshape(1, -1))

        # https://www.quora.com/How-do-you-calculate-the-order-of-two-vectors
        # A is the hor_car vector and B is the opp_car vector
        isOpponentLeft = (hor_car_vec[0] * opp_car_vec[1] - opp_car_vec[0] * hor_car_vec[1]) > 0

        # print "isOpponentLeft {}".format(isOpponentLeft)

        # also reshaping due to indiocyncracies of the cosine similarity function
        cos_sim = cosine_similarity(hor_car_vec.reshape(1, -1), opp_car_vec.reshape(1, -1)).flatten()[0]
        return cos_sim, isOpponentLeft

    def getAngleOfOpponents(self, cars, road):
        self_vec = cars['self'][:2]

        if "others" in cars:
            opponents = cars['others']
            opponentsCount = len(opponents)

            horizon_vec = self.getHorizonVec(road)

            if opponentsCount > 0:
                # cosSims = [
                #     self.getAngleOfOpponent(opponent=opponents[opp_index],
                #                             horizon_vec=horizon_vec, self_vec=self_vec)
                #     for opp_index in range(opponentsCount)]

                self.getAngleOfOpponent(opponent=opponents[0],
                                        horizon_vec=horizon_vec, self_vec=self_vec)

                # for opp_ind, opp in enumerate(opponents):
                #    cosine_similarity( horizon_vec,  )

                # print "cosSims: {}".format(
                #     cosSims
                # )
            else:
                return 0
        else:
            return 0

    def getHorizonVec(self, road):
        middleCoord = self.roadWidth // 2
        roadReference = road[0, middleCoord]
        cameraReference = road[-1, middleCoord]
        return np.array((cameraReference[0], roadReference[1]))

    def howMuchRoadTurning(self, road, action, pixelsThreshold=10, factors=(0.01, 0.1, 0.2, 0.3, 1)):
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
            # straight ahead
            return 1
        else:
            assert roadRef_x != camRef_x
            isRoadTurningLeft = roadRef_x < camRef_x  # is road turning left

            actionPreferred = Action.LEFT if isRoadTurningLeft else Action.RIGHT
            actionNotPreferred = Action.RIGHT if isRoadTurningLeft else Action.LEFT

            # print cosSim
            factors = (1. / cosSim) * np.array(factors)
            # print factors.shape

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

        raise AssertionError  # invalid action detected

    def opponentsBeside(self, grid, action, factors=(0.01, 0.1, 0.2, 0.5, 0.9)):
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
            return 0  # do not let it play a role

        raise AssertionError  # an unexpected action was used

    def distanceFromCentre(self, grid):
        # being in the centre [4 or 5 position] we need the highest value, the lowest value at the edges
        # 4.5 - 0 = 4.5, 4.5 - 9 = -4.5, while 4.5 - 4 = 0.5 and 4.5 - 5 = -0.5
        # if negative means we are on the right, action Left should bring it a larger value, action Right a smaller
        # positive means we are on the left, action Right larger value, action Left smaller
        # action accelerate or break or noop are neutral
        carPos = self.getOurCarPos(grid)
        middlePos = (self.gridWidth - 1) / 2.
        distance = middlePos - carPos
        areWeOnTheRight = distance < 0

        return abs(distance), areWeOnTheRight


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
        print sense.isRoadTurningRight(prevGrid, Action.BRAKE, newGrid)
        print newGrid


    def testTurningLeft():
        prevGrid = sense.generateEmptyGrid()
        newGrid = sense.generateEmptyGrid()
        print prevGrid
        print sense.isRoadTurningLeft(prevGrid, Action.BRAKE, newGrid)
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
