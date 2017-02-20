# from collections import OrderedDict
import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import numpy as np
from store_reward_agent import StoreRewardAgent
# from agent_with_short_orizon_senses import AgentWithShortOrizonSenses
# from agent_with_long_orizon_senses import AgentWithLongOrizonSenses
from agent_with_var_orizon_senses import AgentWithVarOrizonSenses
# from q_dict import Qdict
from q_table import Qtable

if __name__ == "__main__":
    totalEpisodesCount = 100


class QAgent(AgentWithVarOrizonSenses, StoreRewardAgent, Qtable, Agent):
    def __init__(self, rng):
        self.lr_p_param = 1
        assert 0.5 < self.lr_p_param <= 1

        self.debugging = True
        self.gamma = 0.8
        self.computationalTemperature = 10
        self.epsilon = 0.01
        self.actionSelection = self.maxQvalueSelection
        self.initial_state_id = 36  # run agent with senses to find this out
        self.middlefix = "test"

        super(QAgent, self).__init__(rng, howFar=10)

        self.rng = rng

        self.actionById = dict((k, v) for k, v in enumerate(self.getActionsSet()))
        self.idByAction = dict((v, k) for k, v in self.actionById.iteritems())

        print [(k, Action.toString(v)) for k, v in self.actionById.iteritems()]

        self.total_reward = None
        self.curStateId = None
        self.cur_t = None
        self.curAction = None
        self.prevGrid = None
        self.curReward = None
        self.nextStateId = None
        self.episodeCounter = 0

        # from sense import Sense
        # self.anotherSensor = Sense(rng)

    def storeRewardInfo(self):
        # isAnyOfTheBaseClassesShortOrizon = np.any(
        #     [("ShortOrizon".lower() in b.__name__.lower()) for b in QAgent.__bases__])
        # filename = "qagent_" + ("short" if isAnyOfTheBaseClassesShortOrizon else "long") + "_orizon_data"
        totalRewards, rewardStreams = self.getRewardInfo()
        filename = "qagent_%s_orizon_data" % self.middlefix
        np.savez(filename, totalRewards, rewardStreams)
        return filename

    def initialise(self, grid):
        """Called at the beginning of an episode. Use it to construct the initial state."""
        super(QAgent, self).initialise(grid)

        self.total_reward = 0  # Reset the total reward for the episode

        self.curStateId = self.initial_state_id

        self.cur_t = 1

        self.curAction = None
        self.prevGrid = grid
        self.curReward = None
        self.nextStateId = None

        self.episodeCounter += 1

    def run(self, learn, episodes=1, draw=False):
        super(QAgent, self).run(learn, episodes, draw)
        # do something at the end of the run
        super(QAgent, self).appendRewardInfo()

    def getActionsSet(self):
        """including the noop action in our possible actions"""
        return super(QAgent, self).getActionsSet() + [Action.NOOP]

    def getNextLearningRate(self):
        # by step
        # learningRate = 1. / np.power(self.cur_t, self.lr_p_param)
        # self.cur_t += 1
        # return learningRate

        # constant
        # return 0.5

        # by episode
        return 1. / np.power(self.episodeCounter, self.lr_p_param)

    def maxQvalueSelection(self, stateId):
        return self.actionById[
            np.argmax(self.getQbyS(stateId))
        ]

    def softmaxActionSelection(self, stateId):
        Qs = self.getQbyS(stateId)
        exps = np.exp(Qs) / self.computationalTemperature
        summ = np.sum(exps)
        probs = exps / summ

        return self.actionById[np.argmax(probs)]

    def tryRandomActionOr(self, epsilon, callback):
        assert 0. <= epsilon <= 1.
        epsilon = float(epsilon)

        if self.rng.rand() < epsilon:
            return self.getRandomAction()
        else:
            return callback()

    def getRandomAction(self):
        return self.actionById[self.rng.randint(0, len(self.getActionsSet()))]

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        Execute the action and get the received reward signal """

        # IMPORTANT NOTE: 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work

        self.curAction = self.tryRandomActionOr(self.epsilon, lambda: self.actionSelection(self.curStateId))

        self.curReward = self.move(self.curAction)
        # self.curReward = -1 if self.curReward == 0 else self.curReward

        self.total_reward += self.curReward

        super(QAgent, self).act()

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        #print self.anotherSensor.isRoadTurningRight(self.prevGrid, self.curAction, grid)

        self.nextStateId = self.getStateIdBySensing(self.prevGrid, self.curAction, grid)

        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

        self.prevGrid = grid

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
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
        if iteration % (1 if self.debugging else 100) == 0:
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
            print Action.toString(self.curAction)
            print "next state id: %d" % self.nextStateId
            print

        if self.debugging and learn:
            cv2.waitKey(300)
            cv2.imshow("Enduro", self._image)

        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)


if __name__ == "__main__":
    seed = 16011984
    randomGenerator = np.random.RandomState(seed=seed)

    agent = QAgent(rng=randomGenerator)

    agent.run(True, episodes=totalEpisodesCount, draw=True)

    total_rewards, _ = agent.getRewardInfo()
    print total_rewards

    print agent.storeRewardInfo()

    np.save(agent.middlefix + "_bellmanQ", agent.bellmanQ)
