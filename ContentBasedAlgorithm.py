# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:17:17 2019

@author: saumy
"""

from surprise import AlgoBase
from MovieLens import MovieLens
from surprise import PredictionImpossible

import numpy as np
import math

class ContentBasedAlgorithm(AlgoBase):
    def __init__(self,k=40,sim_option={}):
        AlgoBase.__init__(self)
        self.k=k
    
    def similarityBasedOnGenre(self,movie1,movie2,genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy,sumyy = 0,0,0
        for i in range(len(genres1)):
                x = genres1[i]
                y = genres2[i]
                sumxx += x * x
                sumyy += y * y
                sumxy += x * y
        genreSim = sumxy/math.sqrt(sumxx*sumyy)
        return genreSim
    
    def similarityBasedOnYear(self,movie1,movie2,years):
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff / 5.0)
        return sim  
        
    def fit(self,trainset):
        AlgoBase.fit(self, trainset)
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()
        print("Computing content-based similarity matrix")
        # Compute genre distance for every movie combination as a 2x2 matrix
        # create a matrix with all zeros and size of all entries in the dataset
        self.similarities = np.zeros((self.trainset.n_items,self.trainset.n_items))
        for this_rating in range(self.trainset.n_items):
            if (this_rating % 1000 == 0):
                print(this_rating, " of ", self.trainset.n_items)
            for next_rating in range(this_rating+1,self.trainset.n_items):
                this_movieId = int(self.trainset.to_raw_iid(this_rating))
                other_movieId = int(self.trainset.to_raw_iid(next_rating))
                genreSimilarity = self.similarityBasedOnGenre(this_movieId, other_movieId, genres)
                yearSimilarity = self.similarityBasedOnYear(this_movieId, other_movieId, years)

                self.similarities[this_rating, next_rating] = genreSimilarity * yearSimilarity
                self.similarities[next_rating, this_rating] = self.similarities[this_rating, next_rating]
        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i,rating[0]]
            neighbors.append( (genreSimilarity, rating[1]) )
        
        # Extract the top-K most-similar ratings
        k_neighbors = sorted(neighbors,key = lambda x:x[1], reverse=True)[:self.k]
        
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return round(predictedRating,1)
    
   


        
