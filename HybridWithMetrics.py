from MovieLens import MovieLens
from ContentBasedAlgorithm import ContentBasedAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator
from surprise import NormalPredictor
from time import time

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()    
    print('Loading movie ratings..')
    data = ml.loadMovieLensDataset()
    #Compute movie popularity ranks to measure novelty
    rankings = ml.getPopularityRanks()
    return (ml,data,rankings)


np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)
# Just make random recommendations
Random = NormalPredictor()

#Content
ContentKNN = ContentBasedAlgorithm()


#Combine them
Hybrid = HybridAlgorithm([Random, ContentKNN], [0.5, 0.5])
t0=time()
evaluator.AddAlgorithm(Random, "Random")
evaluator.AddAlgorithm(ContentKNN, "ContentKNN")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)

tt=time()-t0
print("time to run %s secs" % round(tt,3))
