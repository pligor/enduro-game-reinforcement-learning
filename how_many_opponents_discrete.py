import numpy as np

class HowManyOpponentsDiscrete(object):

    def __init__(self):
        super(HowManyOpponentsDiscrete, self).__init__()
        self.none = 'none'
        self.afew = 'afew'
        self.alot = 'alot'
        self.collection = [self.none, self.afew, self.alot]

    def getAll(self):
        return self.collection
