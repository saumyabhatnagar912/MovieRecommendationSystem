# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:57:45 2019

@author: Saumya
"""

from MovieLens import MovieLens
from surprise import KNNBasic
from collections import defaultdict
from operator import itemgetter
from time import time

#create object for MovieLens class
ml = MovieLens()
#run the fuction to load the dataset as data
data = ml.loadMovieLensDataset()

#Build a full trainset to be used for calculating the similarity using KNN algorithm
trainSet = data.build_full_trainset()

#Define a similarity measure to estimate a rating. Here we are using cosine similarity.
#since we need to compute similarity based on items, 'user_based':False

sim_options = {'name':'cosine',
               'user_based':False}

model = KNNBasic(sim_options=sim_options)
t0=time()
#to train the algorithm on the trainSet
model.fit(trainSet)

#To generate the similarity matrix
similarityMatrix = model.compute_similarities()

testUser = '0'

#convert the raw ID of the user we want the predictions for into inner ID that can be used by the surprise library
user_inner_id = trainSet.to_inner_uid(testUser)

#Get the default dict of list of ratings for the items the user has already rated
check_user_ratings = trainSet.ur[user_inner_id]

#Get the top K rated items into a list with key as itemID and value as rating
k=40
kNeighbors = sorted(check_user_ratings, key=lambda x: x[1], reverse=True)[:k]
#Get similar items to the item user liked, weighted by rating
similarItems = defaultdict(float)
for itemID, ratings in kNeighbors:
    similarityRow = similarityMatrix[itemID]
    for innerID, score in enumerate(similarityRow):
        similarItems[innerID] += score * (ratings/5.0)

#Build a dictionary of movies the user has already seen, they will be the items in trainSet for which the user has already provided rating
watched={}
for itemID, rating  in trainSet.ur[user_inner_id]:
    watched[itemID] = 1

#Get top-rated items 
count = 0
number = 1
print('Recommended Movies using Item Based Collaborative Filtering:')
for itemID, ratingSum in sorted(similarItems.items(),key=itemgetter(1),reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(number,'-',ml.getMovieName(int(movieID)))
        number += 1
        count += 1
        if (count >= 15):
            break
tt=time()-t0
print("Item based CF Model trained in %s seconds" % round(tt,3))