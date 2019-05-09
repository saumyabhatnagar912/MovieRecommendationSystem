# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:23:00 2019

@author: Saumya
"""
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:
    def __init__(self,data,popularityRanking):
        self.ranking = popularityRanking
        
        #create train and anti test set to be used for prediction using KNNBasleine algorithm
        self.fullTrainingSet = data.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainingSet.build_anti_testset()
        
        #create a training(75%) and test(25%) split. random_state specifies seed for Random Number Generator
        self.trainset, self.testset = train_test_split(data, test_size=0.25,random_state=1)
        
        #To check using Leave-One-Out-Cross-Validation for Top-N recommenders
        LOOCV=LeaveOneOut(n_splits=1, random_state=1) 
        for train,test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test
        #Build anti test set for predictions
        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()
        
        #Calculate similarity to measure diversity using cosine similarity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.fullTrainingSet)
        
        
    def GetFullTrainSet(self):
        return self.fullTrainingSet
    
    def GetFullAntiTestSet(self):
        return self.fullAntiTestSet
    
    #User ratings that are not in the train set. the missing ratings are filled with mean rating of the trainset
    def GetAntiTestSetForUser(self,testSubject):
        trainset = self.fullTrainingSet
        fill = trainset.global_mean
        anti_testSet = []
        user = trainset.to_inner_uid(str(testSubject))
        user_items = set([j for (j,_) in trainset.ur[user]])
        anti_testSet += [(trainset.to_raw_uid(user),trainset.to_raw_iid(i),fill) for i in trainset.all_items() if i not in user_items]
        return anti_testSet
    
    def GetTrainSet(self):
        return self.trainset
    
    def GetTestSet(self):
        return self.testset
    
    def GetLOOCVTrainSet(self):
        return self.LOOCVTrain
    
    def GetLOOCVTestSet(self):
        return self.LOOCVTest
    
    def GetLOOCVAntiTestSet(self):
        return self.LOOCVAntiTestSet
    
    def GetSimilarities(self):
        return self.simsAlgo
    
    def GetPopularityRankings(self):
        return self.ranking
        