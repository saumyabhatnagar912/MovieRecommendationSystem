# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:45:13 2019

@author: Saumya
"""

from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from time import time

from MovieLens import MovieLens

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

    lines1 = spark.read.option("header", "true").csv("C:\\Users\\saumy\\Documents\\RecommendationSystem\\Project\\ratings.csv").rdd
    lines2 = spark.read.option("header", "true").csv("C:\\Users\\saumy\\Documents\\RecommendationSystem\\Project\\ratings_Becky.csv").rdd


    ratingsRDD1 = lines1.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
    ratingsRDD2 = lines2.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
    
    ratings1 = spark.createDataFrame(ratingsRDD1)
    ratings2 = spark.createDataFrame(ratingsRDD2)
    ratings = ratings1.union(ratings2)
    
    (training, test) = ratings.randomSplit([0.8, 0.2])

    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
    t0 = time()
    userRecs = model.recommendForAllUsers(15)
    tt = time() - t0
    print("Model trained in %s seconds" % round(tt,3))
    
    testUser0 = userRecs.filter(userRecs['userId'] == 0).collect()
    
    spark.stop()

    ml = MovieLens()
    ml.loadMovieLensDataset()
        
    for row in testUser0:
        number = 0
        for rec in row.recommendations:
            number += 1
            print(number," - ",ml.getMovieName(rec.movieId))

