import numpy as np


class ActionSelection(object):
    def actionSelection(self, Qs, **kwargs):
        raise NotImplementedError


class EgreedyActionSelection(ActionSelection):
    def __init__(self):
        super(EgreedyActionSelection, self).__init__()

        if hasattr(self, 'getActionsSet'):
            self.__actionById = dict((k, v) for k, v in enumerate(self.getActionsSet()))
            self.__actionsLen = len(self.getActionsSet())
        else:
            raise AssertionError

    def actionSelection(self, Qs, **kwargs):
        if hasattr(self, 'epsilon'):
            return self.tryRandomActionOr(self.epsilon, lambda: self.maxQvalueSelection(Qs))
        else:
            raise AssertionError

    def maxQvalueSelection(self, Qs, **kwargs):
        return self.__actionById[np.argmax(Qs)]

    def tryRandomActionOr(self, epsilon, callback):
        assert 0. <= epsilon <= 1.
        epsilon = float(epsilon)

        if hasattr(self, "rng"):
            if self.rng.rand() < epsilon:
                return self.getRandomAction()
            else:
                return callback()
        else:
            raise AssertionError

    def getRandomAction(self):
        if hasattr(self, "rng"):
            return self.__actionById[self.rng.randint(0, self.__actionsLen)]
        else:
            raise AssertionError


class SoftmaxActionSelection(ActionSelection):
    def __init__(self):
        super(SoftmaxActionSelection, self).__init__()

    @property
    def computationalTemperature(self):
        if hasattr(self, 'computationalTemperatureSpace') and hasattr(self, "episodeCounter"):
            ind = self.episodeCounter - 1
            assert ind >= 0
            compTemp = self.computationalTemperatureSpace[ind]
            #print 'computationalTemperature {}'.format(compTemp)
            return compTemp
        else:
            raise AssertionError

    @staticmethod
    def safeRandomChoice(arr, probs):
        # uncomment if you have a problem with the probs
        # sumP = np.sum(probs)
        # if (1. - sumP) ** 2 > 1e-8:
        #     residual = np.abs(1. - sumP) / len(probs)
        #     if sumP < 1:
        #         probs += residual
        #     else:
        #         probs -= residual
        return np.random.choice(arr, 1, p=probs)[0]

    # def softmaxActionSelection(self, stateId):
    #     Qs = self.getQbyS(stateId)
    #     numerators = np.true_divide(Qs, self.computationalTemperature)
    #     exps = np.exp(numerators)
    #     summ = np.sum(exps)
    #     probs = np.true_divide(exps, summ)
    #     if self.debugging > 0:
    #         print ["%.3f" % p for p in probs]
    #     else:
    #         self.probs_debug = ["%.3f" % p for p in probs], ["%.3f" % p for p in Qs]
    #     # return self.actionById[np.argmax(probs)]
    #     return self.safeRandomChoice(self.getActionsSet(), probs)

    def actionSelection(self, Qs, **kwargs):
        """softmaxActionSelection_computationallySafe"""
        computationalTemperature = kwargs['computationalTemperature'] if 'computationalTemperature' in kwargs else \
            None

        numerators = np.true_divide(Qs, computationalTemperature)
        repeats = np.tile(numerators[np.newaxis], (len(numerators), 1))
        removes = numerators[np.newaxis].T
        denoms = repeats - removes
        expDenoms = np.exp(denoms)
        sumDenoms = np.sum(expDenoms, axis=1)
        probs = 1. / sumDenoms

        if 'onProbs' in kwargs:
            kwargs['onProbs'](probs)

        # return self.actionById[np.argmax(probs)]
        if hasattr(self, 'getActionsSet'):
            return self.safeRandomChoice(self.getActionsSet(), probs)
        else:
            raise AssertionError
