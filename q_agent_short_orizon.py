from collections import OrderedDict
import cv2
#from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import time
import numpy as np
from agent_with_short_orizon_senses import AgentWithShortOrizonSenses

class QAgent(AgentWithShortOrizonSenses):
    def __init__(self, rng):
        super(QAgent, self).__init__(rng)
        # Add member variables to your class here
        self.lr_p_param = 0.51
        assert self.lr_p_param > 0.5 and self.lr_p_param <= 1

        self.gamma = 0.9

        self.actionById = dict((k, v) for k, v in enumerate(self.getActionsSet()))
        self.idByAction = dict((v, k) for k, v in self.actionById.iteritems())

        print [(k, Action.toString(v)) for k, v in self.actionById.iteritems()]

        self.Qshape = (len(self.states), len(self.getActionsSet()))

        self.total_reward = None
        self.bellmanQ = None
        self.curStateId = None
        self.cur_t = None
        self.curAction = None
        self.prevGrid = None
        self.curReward = None
        self.nextStateId = None

    def initialise(self, grid):
        """Called at the beginning of an episode. Use it to construct the initial state."""
        self.total_reward = 0 # Reset the total reward for the episode

        # TODO either set the e greedy policy to something else or give initial value functions
        # here we are giving something larger than zero
        # self.bellmanQ = np.zeros(Qshape)
        self.bellmanQ = OrderedDict(zip(self.getStateIds(), self.rng.rand(self.Qshape[0], self.Qshape[1])))

        self.curStateId = 512  # run keyboard agent with senses to find this out

        self.cur_t = 1

        self.curAction = None
        self.prevGrid = None
        self.curReward = None
        self.nextStateId = None

    def updateQsa(self, stateId, actionId, value):
        self.bellmanQ[stateId][actionId] = value
        return self.bellmanQ[stateId][actionId]

    def getQsa(self, stateId, actionId):
        return self.bellmanQ[stateId][actionId]

    def getQbyS(self, stateId):
        return self.bellmanQ[stateId]

    def getNextLearningRate(self):
        learningRate = 1. / np.power(self.cur_t, self.lr_p_param)
        self.cur_t += 1
        return learningRate

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        Execute the action and get the received reward signal """

        # IMPORTANT NOTE: 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work

        # time.sleep(1)  # seconds
        cv2.waitKey(1000)

        self.curAction = self.actionById[
            np.argmax(self.getQbyS(self.curStateId))
        ]

        print Action.toString(self.curAction)

        self.curReward = self.move(self.curAction)

        self.total_reward += self.curReward

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """

        if self.prevGrid is None:
            self.nextStateId = self.curStateId
        else:
            self.nextStateId = self.getStateIdBySensing(self.prevGrid, self.curAction, grid)

        print "next state id: %d" % self.nextStateId

        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

        self.prevGrid = grid

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        #print self.bellmanQ

        learningRate = self.getNextLearningRate()

        curActionId = self.idByAction[self.curAction]

        curQ = self.getQsa(self.curStateId, curActionId)

        maxNextQ = np.max(self.getQbyS(self.nextStateId))

        updateValue = curQ + learningRate * (self.curReward + self.gamma * maxNextQ - curQ)

        self.updateQsa(self.curStateId, curActionId, updateValue)

        self.curStateId = self.nextStateId

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
    randomGenerator = np.random.RandomState(seed=seed)

    a = QAgent(rng=randomGenerator)

    a.run(True, episodes=1, draw=True)

    print 'Total reward: ' + str(a.total_reward)
