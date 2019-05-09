from MovieLens import MovieLens
from surprise import SVD
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

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# SVD
SVD = SVD(n_epochs = 14, lr_all = 0.005, n_factors = 5)
evaluator.AddAlgorithm(SVD, "SVD")


t0=time()
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
tt=time()-t0
print("SVD Algorithm in %s seconds" % round(tt,3))