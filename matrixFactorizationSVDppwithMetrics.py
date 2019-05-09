from MovieLens import MovieLens
from surprise import SVDpp
from Evaluator import Evaluator
import random
import numpy as np
from time import time

np.random.seed(0)
random.seed(0)

def LoadMovieLensData():
    ml = MovieLens()    
    print('Loading movie ratings..')
    data = ml.loadMovieLensDataset()
    #Compute movie popularity ranks to measure novelty
    rankings = ml.getPopularityRanks()
    return (ml,data,rankings)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

evaluator = Evaluator(evaluationData, rankings)


# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

t0=time()
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
tt=time()-t0
print("SVDpp Algorithm in %s seconds" % round(tt,3))