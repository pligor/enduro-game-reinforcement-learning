import numpy as np


class StoreRewardAgent(object):
    def __init__(self):
        super(StoreRewardAgent, self).__init__()
        self.totalRewards = []
        self.rewardStreams = []
        self.reward_stream = None

    def getRewardInfo(self):
        # return np.array(self.totalRewards), np.array(self.rewardStreams)
        return self.totalRewards, self.rewardStreams

    def initialise(self, grid):
        # do not call super here
        # print "initialize of store reward"

        self.appendRewardInfo()
        self.reward_stream = []

    def appendRewardInfo(self):
        if hasattr(self, 'total_reward'):
            if self.total_reward is not None:
                self.totalRewards.append(self.total_reward)
            if self.reward_stream is not None:
                # self.rewardStreams.append(np.array(self.reward_stream))
                self.rewardStreams.append(self.reward_stream)
        else:
            raise AssertionError

    def act(self):
        if hasattr(self, 'total_reward'):
            self.reward_stream.append(self.total_reward)
        else:
            raise AssertionError


class SaveRewardAgent(StoreRewardAgent):
    def __init__(self):
        super(SaveRewardAgent, self).__init__()
        self.rewardsFilename = None

    def storeRewardInfo(self):
        totalRewards, rewardStreams = self.getRewardInfo()
        filename = self.rewardsFilename
        np.savez(filename, totalRewards, rewardStreams)
        return filename
