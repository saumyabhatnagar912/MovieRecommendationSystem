# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:41:45 2019

@author: saumy
"""

from MovieLens import MovieLens
from ContentBasedAlgorithm import ContentBasedAlgorithm
from Evaluator import Evaluator
from time import time

import random
import numpy as np

np.random.seed(0)
random.seed(0)

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensDataset()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    print("popularity ranks calculated")
    return (ml, data, rankings)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

evaluator = Evaluator(evaluationData, rankings)
t0=time()
contentBased = ContentBasedAlgorithm()
evaluator.AddAlgorithm(contentBased, "ContentBased")

evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
tt=time()-t0
print("Content based CF Model trained in %s seconds" % round(tt,3))
