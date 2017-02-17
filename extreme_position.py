class ExtremePosition:
    far_left = 'far_left'
    far_right = "far_right"
    elsewhere = "elsewhere"

    collection = [far_left, far_right, elsewhere]

    def getAll(self):
        return self.collection
