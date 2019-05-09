# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:46:08 2019

@author: Saumya
"""
from RecommendationMetrics import RecommendationMetrics
from surprise import AlgoBase
#from EvaluationData import EvaluationData

class EvaluatedAlgorithm(AlgoBase):
    def __init__(self,algorithm,name):
        self.algorithm=algorithm
        self.name=name
        
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm
        
    def Evaluate(self,evaluationData, doTopN, n=10, verbose = True):
        metrics={}
        #for MAE and RMSE
        if (verbose):
            print('Evaluating Accuracy...')
            self.algorithm.fit(evaluationData.GetTrainSet())
            predictions = self.algorithm.test(evaluationData.GetTestSet())
            metrics["MAE"] = RecommendationMetrics.MAE(predictions)
            metrics["RMSE"] = RecommendationMetrics.RMSE(predictions)
            
        if (doTopN):
            #Evaluate top 10 predictions with leave-one-out testing
            if (verbose):
                print("Evaluating Top-N with leave one out approach...")
            self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())
            #Building prediction for all the ratings that are not in train set
            allPredictions = self.algorithm.test(evaluationData.GetLOOCVAntiTestSet())
            #Compute top-10 records for each user
            topNPredicted = RecommendationMetrics.GetTopN(allPredictions,n)
            if (verbose):
                print("Computing hit rank and rank metrics...")
            #Check how often we recommend a move the user has actually rated
            metrics["HR"] = RecommendationMetrics.HitRate(topNPredicted,leftOutPredictions)
            metrics["ARHR"] = RecommendationMetrics.AvergeReciprocalHitRank(topNPredicted,leftOutPredictions)
            if (verbose):
                print("Analyzing diversity and novelty")
            #Measure diversity of recommendation
            metrics["Diversity"] = RecommendationMetrics.Diversity(topNPredicted,evaluationData.GetSimilarities())
            metrics["Novelty"] = RecommendationMetrics.Novelty(topNPredicted,evaluationData.GetPopularityRankings())
        if (verbose):
            print('Analysis Complete!')
        
        return metrics
    
    