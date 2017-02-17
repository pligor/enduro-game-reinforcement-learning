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

        #TODO
        self.bellmanQ = np.zeros((len(states), len(self.getActionsSet())))

        print "bellman q initially"
        print self.bellmanQ

        self.curState = 0 #initially it is right (right is 0, left is 1)

        self.gamma = 0.9

    def isLeft(self, grid):
        row = grid[0, :]
        half_ind = int(len(row)/2)
        return np.all(row[:half_ind] != 2)

    # def getActionById(self):
    #     """here the convention is that each action in the set"""
    #     pass
    def getQsa(self, stateId, actionId):
        """it is trivial in this example but might not be like this in the future"""
        return self.bellmanQ[stateId, actionId]

    def getQbyS(self, stateId):
        """this is very simplified since the rows of bellmanQ are the state id"""
        return self.bellmanQ[stateId]

    def __init__(self):
        super(QAgent, self).__init__()
        # Add member variables to your class here

        print [Action.toString(a) for a in self.getActionsSet()]

        self.actionById = dict( (k, v) for k, v in enumerate(self.getActionsSet()) )
        self.idByAction = dict( (v, k) for k, v in self.actionById.iteritems() )

        print self.idByAction

        #print [(k, Action.toString(v)) for k, v in self.actionById.iteritems()]

        self.init()

    def initialise(self, grid):
        """Called at the beginning of an episode. Use it to construct the initial state."""
        self.init()

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work
        # self.total_reward += self.move(action)

        self.curAction = self.actionById[
            np.argmax(self.getQbyS(self.curState))
        ]

        print Action.toString(self.curAction)

        self.curReward = self.move(self.curAction)

        self.total_reward += self.curReward

        time.sleep(2) #seconds

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """

        self.nextState = int(self.isLeft(grid))

        #print self.nextState
        #print grid

        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        #self.curReward
        #self.curState
        #self.nextState

        curQ = self.getQsa(self.curState, )

        self.curState = self.nextState
        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)

if __name__ == "__main__":
    a = QAgent()
    a.run(True, episodes=2, draw=True)
    print 'Total reward: ' + str(a.total_reward)
