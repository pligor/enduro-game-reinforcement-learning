import numpy as np

class HowManyOpponents(object):

    def __init__(self, maxcount=3):
        super(HowManyOpponents, self).__init__()
        self.maxcount = maxcount
        self.many = 'many'
        self.collection = np.concatenate((np.arange(maxcount+1).astype(np.str), np.array([self.many])))

    def getAll(self):
        return self.collection
