# import sys

# sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
from sense import Sense
import numpy as np
from enduro_data_types import tuple_dt
from agent_with_short_orizon_senses import AgentWithShortOrizonSenses

class KeyboardAgent(AgentWithShortOrizonSenses):
    def __init__(self, rng):
        super(KeyboardAgent, self).__init__(rng)
        # Add member variables to your class here
        self.total_reward = 0

        self.prevGrid = None

        self.curAction = None

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0

        self.prevGrid = None
        self.curAction = None

        print "enduro_states_short_horizon: %d" % len(self.states)

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        key = cv2.waitKey(400)  # 400 for normal play, 5000 for step by step
        action = Action.NOOP
        if chr(key & 255) == 'a':
            action = Action.LEFT
        if chr(key & 255) == 'd':
            action = Action.RIGHT
        if chr(key & 255) == 'w':
            action = Action.ACCELERATE
        if chr(key & 255) == 's':
            action = Action.BREAK

        self.curAction = action

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT or Action.ACCELERATE
        # Do not use plain integers between 0 - 3 as it will not work
        self.total_reward += self.move(self.curAction)

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        grid -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """

        if self.prevGrid is not None:
            stateId = self.getStateIdBySensing(self.prevGrid, self.curAction, grid)
            print stateId

        #print grid

        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

        self.prevGrid = grid

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        # Show the latest game frame
        cv2.imshow("Enduro", self._image)


if __name__ == "__main__":
    seed = 16011984
    rng = np.random  # .RandomState(seed=seed)

    a = KeyboardAgent(rng)

    a.run(False, episodes=1, draw=True)
    print 'Total reward: ' + str(a.total_reward)
