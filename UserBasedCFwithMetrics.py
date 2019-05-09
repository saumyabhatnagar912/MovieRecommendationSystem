# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:04:01 2019

@author: Saumya
"""

from MovieLens import MovieLens
from Evaluator import Evaluator
from surprise import KNNBasic
from time import time

def LoadMovieLensData():
    ml = MovieLens()
    print('Loading movie ratings..')
    data = ml.loadMovieLensDataset()
    #Compute movie popularity ranks to measure novelty
    rankings = ml.getPopularityRanks()
    return (ml,data,rankings)



#Load the common data set for the recommender algorithms
(ml,evaluationData,rankings) = LoadMovieLensData()

#create an evaluator which is an object of Evaluator class
evaluator = Evaluator(evaluationData,rankings)

t0= time()
#User-Based KNN
sim_options_user = {'name':'cosine','user_based':True}
userKNN = KNNBasic(sim_options = sim_options_user)
evaluator.AddAlgorithm(userKNN,"User KNN")

evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
tt=time()-t0
print("User based CF Model trained in %s seconds" % round(tt,3))