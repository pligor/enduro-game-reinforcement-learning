from __future__ import division

class Constrainer(object):
    def __init__(self, new_min=-1, new_max=+1):
        super(Constrainer, self).__init__()
        self.new_min = new_min
        self.new_max = new_max
        self.new_range = self.new_max - self.new_min

        # start with the assumption that could be the same as our original ones
        self.old_min = new_min
        self.old_max = new_max

    @property
    def old_range(self):
        return self.old_max - self.old_min

    @old_range.setter
    def old_range(self, value):
        self.old_max = max(self.old_max, value)
        self.old_min = min(self.old_min, value)

    def constrain(self, old_value):
        """http://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio"""
        old_range = self.old_range

        if old_range == 0:
            return (self.new_min + self.new_max) / 2
        else:
            return (((old_value - self.old_min) * self.new_range) / old_range) + self.new_min
