class RoadCategory:
    turn_left = 'turn_left'
    turn_right = "turn_right"
    straight_ahead = "straight_ahead"

    collection = [turn_left, turn_right, straight_ahead]

    def getAll(self):
        return self.collection
