import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import time
import numpy as np
#import matplotlib.pyplot as plt #it's not working

class RandomAgent(Agent):
    def getRewardInfo(self):
        return np.array(self.totalRewards), np.array(self.rewardStreams)

    def getActionsSet(self):
        """including the noop action in our possible actions"""
        return super(RandomAgent, self).getActionsSet() + [Action.NOOP]

    def __init__(self, rng):
        super(RandomAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = None
        self.reward_stream = None
        self.rng = rng
        #self.counter = 0
        print [Action.toString(a) for a in self.getActionsSet()]
        print type(self.getActionsSet())
        #print len(self.getActionsSet())

        self.totalRewards = []
        self.rewardStreams = []

    def initialise(self, grid):
        if self.total_reward is not None:
            self.totalRewards.append(self.total_reward)
        if self.reward_stream is not None:
            self.rewardStreams.append(np.array(self.reward_stream))

        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0
        print "new episode"
        self.reward_stream = []

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        actionSet = self.getActionsSet()

        actionInd = self.rng.randint(0, len(actionSet))
        #print actionInd

        #action = Action.NOOP
        action = actionSet[actionInd]
        #print Action.toString(action)

        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work

        cur_reward = self.move(action)

        self.reward_stream.append(cur_reward)

        self.total_reward += cur_reward

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
    rng = np.random.RandomState(seed=seed)
    agent = RandomAgent(rng=rng)

    agent.run(True, episodes=100, draw=True)
    print 'Total reward: ' + str(agent.total_reward)

    totalRewards, rewardStreams = agent.getRewardInfo()

    print totalRewards
    meanTotalRewards = np.mean(totalRewards)
    varTotalRewards = np.var(totalRewards)
    print "Mean of Total Rewards: %f" % meanTotalRewards
    print "Variance to Total Rewards: %f" % varTotalRewards

    np.savez('random_data', totalRewards, rewardStreams)
