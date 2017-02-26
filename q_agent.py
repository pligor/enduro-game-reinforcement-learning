# from collections import OrderedDict
from __future__ import division
import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import numpy as np
from store_reward_agent import StoreRewardAgent
from agent_with_boeing_senses import AgentWithBoeingSenses
from agent_with_var_orizon_senses import AgentWithVarOrizonSenses
# from q_dict import Qdict
from q_table import Qtable, Qcase
from skopt.space.space import Integer, Real
from skopt import gp_minimize
from os.path import isfile

if __name__ == "__main__":
    totalEpisodesCount = 100
    seed = 16011984

class QAgent(AgentWithBoeingSenses, StoreRewardAgent, Qtable, Agent):
    def __init__(self, rng, computationalTemperature):
        self.lr_p_param = 0.501
        assert 0.5 < self.lr_p_param <= 1

        self.epsilon = 0.
        self.actionSelection = self.softmaxActionSelection_computationallySafe
        #self.computationalTemperature = 5e-3  # small more like max, large more like random
        self.computationalTemperature = computationalTemperature  # small more like max, large more like random

        #self.epsilon = 0.01
        #self.actionSelection = self.maxQvalueSelection

        self.debugging = 0 #zero for actual run
        self.gamma = 0.8
        self.initial_state_id = 5184  # run agent with senses to find this out
        self.middlefix = "boeing"
        #self.initialQ = Qcase.ZERO

        def changeQtable(table):
            table[:, 0] = 10
            #table[:, 1:4] = 0
            return table

        self.initialQ = changeQtable

        self.rng = rng

        # super(QAgent, self).__init__(rng, howFar=10)
        super(QAgent, self).__init__(rng)

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

    def run(self, learn, episodes=1, draw=False):
        super(QAgent, self).run(learn, episodes, draw)
        # do something at the end of the run
        super(QAgent, self).appendRewardInfo()

    def getActionsSet(self):
        """including the noop action in our possible actions"""
        if hasattr(self, "allActionSet") and self.allActionSet is not None:
            return self.allActionSet
        else:
            self.allActionSet = super(QAgent, self).getActionsSet() + [Action.NOOP]
            return self.allActionSet

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

    @staticmethod
    def safeRandomChoice(arr, probs):
        #uncomment if you have a problem with the probs
        # sumP = np.sum(probs)
        # if (1. - sumP) ** 2 > 1e-8:
        #     residual = np.abs(1. - sumP) / len(probs)
        #     if sumP < 1:
        #         probs += residual
        #     else:
        #         probs -= residual
        return np.random.choice(arr, 1, p=probs)[0]

    def softmaxActionSelection(self, stateId):
        Qs = self.getQbyS(stateId)
        numerators = np.true_divide(Qs, self.computationalTemperature)
        exps = np.exp(numerators)
        summ = np.sum(exps)
        probs = np.true_divide(exps, summ)
        if self.debugging > 0:
            print ["%.3f" % p for p in probs]
        else:
            self.probs_debug = ["%.3f" % p for p in probs], ["%.3f" % p for p in Qs]
        # return self.actionById[np.argmax(probs)]
        return self.safeRandomChoice(self.getActionsSet(), probs)

    def softmaxActionSelection_computationallySafe(self, stateId):
        Qs = self.getQbyS(stateId)
        numerators = np.true_divide(Qs, self.computationalTemperature)
        repeats = np.tile(numerators[np.newaxis], (len(numerators), 1))
        removes = numerators[np.newaxis].T
        denoms = repeats - removes
        expDenoms = np.exp(denoms)
        sumDenoms = np.sum(expDenoms, axis=1)
        probs = 1. / sumDenoms
        if self.debugging > 0:
            print ["%.3f" % p for p in probs]
        else:
            self.probs_debug = ["%.3f" % p for p in probs], ["%.1f" % p for p in Qs]
        # return self.actionById[np.argmax(probs)]
        return self.safeRandomChoice(self.getActionsSet(), probs)

    def tryRandomActionOr(self, epsilon, callback):
        assert 0. <= epsilon <= 1.
        epsilon = float(epsilon)

        if self.rng.rand() < epsilon:
            return self.getRandomAction()
        else:
            return callback()

    def getRandomAction(self):
        return self.actionById[self.rng.randint(0, len(self.getActionsSet()))]

    @staticmethod
    def areOpponents(grid):
        return np.sum(grid == 1)

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

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        Execute the action and get the received reward signal """

        # IMPORTANT NOTE: 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work

        self.curAction = self.tryRandomActionOr(self.epsilon, lambda: self.actionSelection(self.curStateId))

        self.curReward = self.move(self.curAction)
        #self.curReward = -0.01 if self.curReward == 0 else self.curReward

        self.total_reward += self.curReward

        super(QAgent, self).act()

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # print self.anotherSensor.isRoadTurningRight(self.prevGrid, self.curAction, grid)

        self.nextStateId = self.getStateIdBySensing(self.prevGrid, self.curAction, grid)

        # Visualise the environment grid
        if self.debugging > 0:
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
        if iteration % (100 if self.debugging == 0 else 1) == 0:
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
            print Action.toString(self.curAction)
            print "next state id: %d" % self.nextStateId
            if self.actionSelection == self.softmaxActionSelection_computationallySafe:
                print self.probs_debug
            print

        if self.debugging > 0 and learn:
            cv2.waitKey(self.debugging)
            cv2.imshow("Enduro", self._image)

        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)


if __name__ == "__main__":
    def mymain(computationalTemperature):
        print "computationalTemperature"
        print computationalTemperature

        randomGenerator = np.random.RandomState(seed=seed)

        agent = QAgent(rng=randomGenerator, computationalTemperature=computationalTemperature)

        agent.run(True, episodes=totalEpisodesCount, draw=True)

        total_rewards, _ = agent.getRewardInfo()
        print total_rewards

        print agent.storeRewardInfo()

        np.save(agent.middlefix + "_bellmanQ", agent.bellmanQ)

        return np.mean(total_rewards)


    def objective(params):  # Here we define the metric we want to minimise
        (computationalTemperature, ) = params

        meanOfTotalRewards = mymain(computationalTemperature)

        # We want to maximise average accuracy, i.e. minimise minus average accuracy
        return -meanOfTotalRewards

    space = [Real(1e-4, 1e-1)]

    bestComputationTempFilename = 'best_comp_temp.npy'

    if isfile(bestComputationTempFilename):
        (bestComputationTemperature, ) = np.load(bestComputationTempFilename)
    else:
        #beware this runs more than 15 hours for the many states case
        res_gp = gp_minimize(
            func=objective,  # function that we wish to minimise
            dimensions=space,  # the search space for the hyper-parameters
            #x0=x0,  # inital values for the hyper-parameters
            n_calls=25,  # number of times the function will be evaluated
            random_state=seed,  # random seed
            n_random_starts=1,  # before we start modelling the optimised function with a GP Regression
            # model, we want to try a few random choices for the hyper-parameters.
            kappa=1.9  # trade-off between exploration vs. exploitation.
        )
        print "best param"
        print res_gp.x
        np.save(bestComputationTempFilename, res_gp.x)

    print "bestComputationTemperature"
    print bestComputationTemperature

    meanTotalRewards = mymain(bestComputationTemperature)
    print "meanTotalRewards"
    print meanTotalRewards
