# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:45:13 2019

@author: Saumya
"""
from surprise import accuracy
from collections import defaultdict
import itertools

class RecommendationMetrics:
    
#MAE and RMSE calculated using accuracy module from surprise lib
    def MAE(predictions):
        return accuracy.mae(predictions,verbose=False)
    
    def RMSE(predictions):
        return accuracy.rmse(predictions,verbose=False)
        
    def GetTopN(predictions, n=10, minimumRating=4.0):
        topN = defaultdict(list)
        #Of all the predictions, take those above a threshold and store them in topN dictionary of list
        #Prediction(uid='raw user id', iid='raw item id', r_ui=true rating, est=estimated rating, details={additional details})
        for userID, movieID,actualRating,estimatedRating,_ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID),estimatedRating))
        #sort the dictionary based on ranking, keep the top n sorted items for each user in the dictionary and remove lower ranking items
        for userID, ratings in topN.items():
            ratings.sort(key=lambda x:x[1],reverse = True)
            topN[int(userID)] = ratings[:n]
            
        return topN
    
    #Hit Rate is calculated using Leave One Out approach
    def HitRate(topNPredicted,leftOutPredictions):
        hits = 0
        total = 0
        #For each left out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieId = leftOut[1]
            hit = False
            #Check if the predicted movie is in the top 10 for this user
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if(int(leftOutMovieId) == int (movieID)):
                    hit = True
                    break
            if (hit):
                hits += 1
                
            total +=1
        hitRate = hits/total
        return hitRate
    
    def AvergeReciprocalHitRank(topNPredicted,leftOutPredictions):
        summation = 0
        total = 0
        #For each left out rating
        for userID,leftOutMovieID, actualRating, estimatedRating,_ in leftOutPredictions:
            hitRank = 0
            rank = 0
            #if the movie is in top n predicted rating list
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank += 1
                if (int(leftOutMovieID) == int(movieID)):
                    hitRank = rank
                    break
            
            if (hitRank>0):
                summation += 1.0/hitRank
                
            total += 1
            aRHR = summation/total
        
        return aRHR
        
    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userId in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userId],2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerId1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerId2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerId1][innerId2]
                total += similarity
                n += 1
        sim = total/n
        diversity = (1 - sim)
        return diversity
    
    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userId in topNPredicted.keys():
            for rating in topNPredicted[userId]:
                movieId = rating[0]
                rank = rankings[movieId]
                total += rank
                n += 1
        novelty = total/n
        return novelty