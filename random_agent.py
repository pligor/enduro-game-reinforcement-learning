import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import time
import numpy as np
from store_reward_agent import StoreRewardAgent
#import matplotlib.pyplot as plt #it's not working

class RandomAgent(StoreRewardAgent, Agent):
    def getActionsSet(self):
        """including the noop action in our possible actions"""
        return super(RandomAgent, self).getActionsSet() + [Action.NOOP]

    def __init__(self, rng):
        super(RandomAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = None
        self.rng = rng
        self.curReward = None
        print [Action.toString(a) for a in self.getActionsSet()]
        print type(self.getActionsSet())

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct the initial state."""
        super(RandomAgent, self).initialise(grid)

        # Reset the total reward for the episode
        self.total_reward = 0
        print "new episode"
        self.curReward = None

    def run(self, learn, episodes=1, draw=False):
        super(RandomAgent, self).run(learn, episodes, draw)
        #do something at the end of the run
        super(RandomAgent, self).appendRewardInfo()

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        actionSet = self.getActionsSet()

        actionInd = self.rng.randint(0, len(actionSet))

        action = actionSet[actionInd]
        #print Action.toString(action)

        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work

        self.curReward = self.move(action)

        self.total_reward += self.curReward

        super(RandomAgent, self).act()

        #cv2.waitKey(1000)

    @staticmethod
    def isLeft(grid):
        row = grid[0, :]
        half_ind = int(len(row)/2)
        return np.all(row[:half_ind] != 2)

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        grid -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """

        #print grid[0, :]
        #print self.isLeft(grid)

        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        #self.counter += 1
        #print "counter: %d" % self.counter
        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        if iteration % 100 == 0: #show all specific
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)

        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)

if __name__ == "__main__":
    seed = 16011984
    randomGenerator = np.random.RandomState(seed=seed)
    agent = RandomAgent(rng=randomGenerator)

    agent.run(True, episodes=100, draw=True)

    totalRewards, rewardStreams = agent.getRewardInfo()

    print len(rewardStreams)
    print totalRewards
    meanTotalRewards = np.mean(totalRewards)
    varTotalRewards = np.var(totalRewards)
    print "Mean of Total Rewards: %f" % meanTotalRewards
    print "Variance to Total Rewards: %f" % varTotalRewards

    np.savez('random_data', totalRewards, rewardStreams)
