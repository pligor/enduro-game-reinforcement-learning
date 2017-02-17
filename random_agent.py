import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import time
import numpy as np
#import matplotlib.pyplot as plt #it's not working

class RandomAgent(Agent):
    def __init__(self, rng):
        super(RandomAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.rng = rng
        #self.counter = 0
        print [Action.toString(a) for a in self.getActionsSet()]
        #print len(self.getActionsSet())

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0
        print "new episode"

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

        self.total_reward += self.move(action)
        time.sleep(1)

    def isLeft(self, grid):
        row = grid[0, :]
        half_ind = int(len(row)/2)
        return np.all(row[:half_ind] != 2)

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        grid -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        print grid[0, :]
        print self.isLeft(grid)
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
        if np.allclose(iteration % 500, 0): #show all specific
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)

        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)

if __name__ == "__main__":
    seed = 16011984
    rng = np.random.RandomState(seed=seed)
    agent = RandomAgent(rng=rng)
    #agent.run(True, episodes=2, draw=True)
    size = 1
    totalRewards = -1000 * np.ones(size, dtype=np.int) #minus zero to show that this is something invalid

    for i in range(size):
        agent.run(True, episodes=1, draw=True)
        print 'Total reward: ' + str(agent.total_reward)
        totalRewards[i] = agent.total_reward

    #print np.mean()
    print totalRewards
    print "Mean of Total Rewards: %f" % np.mean(totalRewards) #
    print "Variance to Total Rewards: %f" % np.var(totalRewards) #

    #fig = plt.figure()
    #plt.plot(np.arange(1, size+1), totalRewards)
    #plt.show()
    np.save('random_data', totalRewards)
