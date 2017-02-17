import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import time
import numpy as np


class QAgent(Agent):
    def init(self):
        # Reset the total reward for the episode
        self.total_reward = 0

        states = np.zeros(2)

        Qshape = (len(states), len(self.getActionsSet()))

        # TODO either set the e greedy policy to something else or give initial value functions
        #here we are giving something larger than zero
        #self.bellmanQ = np.zeros(Qshape)
        self.bellmanQ = self.rng.rand(Qshape[0], Qshape[1])

        print "bellman q initially"
        print self.bellmanQ

        self.curStateId = 0  # initially it is right (right is 0, left is 1)

        self.gamma = 0.9

        self.cur_t = 1

    def isLeft(self, grid):
        row = grid[0, :]
        half_ind = int(len(row) / 2)
        return np.all(row[:half_ind] != 2)

    # def getActionById(self):
    #     """here the convention is that each action in the set"""
    #     pass
    def updateQsa(self, stateId, actionId, value):
        self.bellmanQ[stateId, actionId] = value
        return self.bellmanQ[stateId, actionId]

    def getQsa(self, stateId, actionId):
        """it is trivial in this example but might not be like this in the future"""
        return self.bellmanQ[stateId, actionId]

    def getQbyS(self, stateId):
        """this is very simplified since the rows of bellmanQ are the state id"""
        return self.bellmanQ[stateId]

    def __init__(self, rng):
        super(QAgent, self).__init__()
        # Add member variables to your class here
        self.actionById = dict((k, v) for k, v in enumerate(self.getActionsSet()))
        self.idByAction = dict((v, k) for k, v in self.actionById.iteritems())

        print [(k, Action.toString(v)) for k, v in self.actionById.iteritems()]

        self.rng = rng

        self.lr_p_param = 0.51
        assert self.lr_p_param > 0.5 and self.lr_p_param <= 1

        self.init()

    def getNextLearningRate(self):
        learningRate = 1. / np.power(self.cur_t, self.lr_p_param)
        self.cur_t += 1
        return learningRate

    def initialise(self, grid):
        """Called at the beginning of an episode. Use it to construct the initial state."""
        self.init()

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work
        # self.total_reward += self.move(action)

        self.curAction = self.actionById[
            np.argmax(self.getQbyS(self.curStateId))
        ]

        print Action.toString(self.curAction)

        self.curReward = self.move(self.curAction)

        self.total_reward += self.curReward

        time.sleep(1)  # seconds

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """

        self.nextStateId = int(self.isLeft(grid))

        print "next state id: %d" % self.nextStateId

        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        print self.bellmanQ

        learningRate = self.getNextLearningRate()

        curActionId = self.idByAction[self.curAction]

        curQ = self.getQsa(self.curStateId, curActionId)

        maxNextQ = np.max(self.getQbyS(self.nextStateId))

        updateValue = curQ + learningRate*(self.curReward + self.gamma * maxNextQ - curQ)

        self.updateQsa(self.curStateId, curActionId, updateValue)

        self.curStateId = self.nextStateId

        return

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)
        print


if __name__ == "__main__":
    seed = 16011984
    rng = np.random.RandomState(seed=seed)

    a = QAgent(rng=rng)

    a.run(True, episodes=1, draw=True)

    print 'Total reward: ' + str(a.total_reward)
