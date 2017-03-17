from enduro.action import Action
import cv2

class KeyboardControl(object):
    def __init__(self):
        super(KeyboardControl, self).__init__()

    def selectActionByKeyboard(self):
        key = cv2.waitKey(400)  # 300 for normal play, 5000 for step by step
        action = Action.NOOP
        if chr(key & 255) == 'a':
            action = Action.LEFT
        if chr(key & 255) == 'd':
            action = Action.RIGHT
        if chr(key & 255) == 'w':
            action = Action.ACCELERATE
        if chr(key & 255) == 's':
            action = Action.BRAKE

        return action
