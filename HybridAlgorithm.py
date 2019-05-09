from surprise import AlgoBase
from surprise import NormalPredictor

class HybridAlgorithm(AlgoBase):
    def __init__(self, algorithms, weights,sim_options={}):
        AlgoBase.__init__(self)
        self.algorithms=algorithms
        self.weights=weights

    def fit(self,trainset):
        AlgoBase.fit(self,trainset)
        for algo in self.algorithms:
            algo.fit(trainset)
        
        return self

    def estimate(self,u,i):
        scores_total = 0
        weights_total = 0
        for index in range (len(self.algorithms)):
            scores_total += self.algorithms[index].estimate(u,i) * self.weights[index]
            weights_total += self.weights[index]

        return scores_total/weights_total
