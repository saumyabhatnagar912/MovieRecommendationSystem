# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:31:45 2019

@author: Saumya
"""
from EvaluatedAlgorithm import EvaluatedAlgorithm
from EvaluationData import EvaluationData

class Evaluator:
    algorithms = []
    
    def __init__(self,dataset,rankings):
        ed=EvaluationData(dataset,rankings)
        self.dataset = ed
        
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm,name)
        self.algorithms.append(alg)

    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

        # Print results
        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "ARHR", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                       name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["ARHR"],
                                       metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
                
        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if (doTopN):
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better." )
            print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")
    
    def SampleTopNRecs(self, ml, testSubject=0, k=10):
        for algo in self.algorithms:
            print("\nUsing ", algo.GetName())
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            print("Computing Recommendations")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
            predictions = algo.GetAlgorithm().test(testSet)
            recommendations = []
            print("\nFor user ", testSubject, " we recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID,estimatedRating))
                recommendations.sort(key = lambda x:x[1],reverse = True)
            number=0
            for ratings in recommendations[:15]:
                number += 1
                print(number," - ",ml.getMovieName(ratings[0]), round(ratings[1],1))