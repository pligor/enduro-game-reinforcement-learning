import numpy as np

class StoreRewardAgent(object):
    def __init__(self):
        super(StoreRewardAgent, self).__init__()
        self.totalRewards = []
        self.rewardStreams = []
        self.reward_stream = None

    def getRewardInfo(self):
        #return np.array(self.totalRewards), np.array(self.rewardStreams)
        return self.totalRewards, self.rewardStreams

    def initialise(self, grid):
        #do not call super here
        #print "initialize of store reward"

        self.appendRewardInfo()
        self.reward_stream = []

    def appendRewardInfo(self):
        assert hasattr(self, 'total_reward')

        if self.total_reward is not None:
            self.totalRewards.append(self.total_reward)
        if self.reward_stream is not None:
            #self.rewardStreams.append(np.array(self.reward_stream))
            self.rewardStreams.append(self.reward_stream)

    def act(self):
        assert hasattr(self, 'total_reward')

        self.reward_stream.append(self.total_reward)

