# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:34:30 2019

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
#similarities will be computed between users or between items.since we need to compute similarity based on users, 'user_based':True

sim_options = {'name':'cosine',
               'user_based':True}

t0=time()
#using the similarity model as KNNBasic 
model = KNNBasic(sim_options=sim_options)
#to train the algorithm on the trainSet
model.fit(trainSet)

#To generate the similarity matrix
similarityMatrix = model.compute_similarities()

testUser = '0'

#Get Top N similar users to our test user
user_inner_id = trainSet.to_inner_uid(testUser)

similarityRow = similarityMatrix[user_inner_id]
similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID!=user_inner_id):
        similarUsers.append((innerID,score))
k=40
kNeighbors = sorted(similarUsers, key=lambda x: x[1], reverse=True)[:k]

#Get the items they rated, and add the rating for each item weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    userRatings = trainSet.ur[innerID]
    for rating in userRatings:
        #here rating[0] is the userID
        candidates[rating[0]] += userSimilarityScore*(rating[1]/5.0)
        
        
#Build a dictionary of movies the testUser has already seen. That would mean movies for which testUser's rating is available in the trainSet
watched={}
for itemID, _ in trainSet.ur[user_inner_id]:
    watched[itemID] = 1

#Get top 10 rated movies from similar users:
count = 0
number = 1
print('Recommended Movies using User Based Collaborative Filtering:')
for itemID, _ in sorted(candidates.items(),key=itemgetter(1),reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(number,'-',ml.getMovieName(int(movieID)))
        number += 1
        count += 1
        if (count >= 15):
            break

tt=time()-t0
print("User based CF Model trained in %s seconds" % round(tt,3))