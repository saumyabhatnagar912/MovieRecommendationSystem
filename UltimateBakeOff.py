from MovieLens import MovieLens
from surprise import SVD, SVDpp
from Evaluator import Evaluator
from ContentBasedAlgorithm import ContentBasedAlgorithm
from surprise import KNNBasic
import numpy as np
import random
from time import time
from surprise import NormalPredictor
from HybridAlgorithm import HybridAlgorithm
from surprise.model_selection import GridSearchCV

import random
import numpy as np

np.random.seed(0)
random.seed(0)

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

t0=time()
evaluator = Evaluator(evaluationData, rankings)

#Content-based
ContentKNN = ContentBasedAlgorithm()
evaluator.AddAlgorithm(ContentKNN, "ContentBased")

#User-Based 
sim_options_user = {'name':'cosine','user_based':True}
userKNN = KNNBasic(sim_options = sim_options_user)
evaluator.AddAlgorithm(userKNN,"UserBased")


#Item-Based
sim_options_item = {'name':'cosine','user_based':False}
itemKNN = KNNBasic(sim_options = sim_options_item)
evaluator.AddAlgorithm(itemKNN,"ItemBased")


# SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")


# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")


#Hybrid
#Random and Content-based
Random = NormalPredictor()
Hybrid = HybridAlgorithm([Random, ContentKNN], [0.5, 0.5])
evaluator.AddAlgorithm(Hybrid,'Hybrid')


evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)

tt=time()-t0
print("Ultimate Algorithm bake-off in %s seconds" % round(tt,3))
