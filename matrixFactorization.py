from MovieLens import MovieLens
import numpy as np
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV
import random

def LoadMovieLensData():
    ml = MovieLens()    
    print('Loading movie ratings..')
    data = ml.loadMovieLensDataset()
    #Compute movie popularity ranks to measure novelty
    rankings = ml.getPopularityRanks()
    return (ml,data,rankings)

#set up a see to get consistant results
np.random.seed(0)
random.seed(0)

#Load the common data set for the recommender algorithms
(ml,evaluationData,rankings) = LoadMovieLensData()

print("searching for the best parameters for svd...")

param_grid = {'n_epochs':[14,14],'lr_all':[0.005,0.005], 'n_factors':[10,5]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluationData)

print("Best RMSE score attained: ", gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print("Best parameters: ",gs.best_params['rmse'])

evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']
SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
evaluator.AddAlgorithm(SVDtuned, "SVD - Tuned")

SVDUntuned = SVD()
evaluator.AddAlgorithm(SVDUntuned, "SVD - Untuned")

evaluator.Evaluate(True)
evaluator.SampleTopNRecs(ml)
