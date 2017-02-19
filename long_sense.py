import numpy as np
from enduro.action import Action
from road_category import RoadCategory
from extreme_position import ExtremePosition
from sense import Sense
from how_many_opponents import HowManyOpponents

class LongSense(Sense):
    def __init__(self, rng):
        super(LongSense, self).__init__(rng)

    def countOpponents(self, grid, left_boolean):
        assert self.gridWidth % 2 == 0
        targetArea = grid[1:, :self.gridWidth/2] if left_boolean else grid[1:, self.gridWidth/2:]
        count = np.sum(targetArea == 1)
        howManyOpponents = HowManyOpponents()
        if count > howManyOpponents.maxcount:
            return howManyOpponents.many
        else:
            return str(count)


if __name__ == "__main__":
    seed = 16011984
    rng = np.random  # .RandomState(seed=seed)
    sense = LongSense(rng=rng)

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

