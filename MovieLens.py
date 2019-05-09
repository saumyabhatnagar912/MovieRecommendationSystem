# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:05:53 2019

@author: Saumya
"""
import csv
import re
import pandas as pd
from surprise import Reader
from surprise import Dataset
from collections import defaultdict
import os
import sys

class MovieLens:
    os.chdir(os.path.dirname(sys.argv[0]))    
    rating_file_location = 'ratings.csv'
    movies_file_location = 'movies.csv'
    my_rating_file_location = 'ratings_Becky.csv'
    new_rating_csv = 'ratings-Data.csv'
    movieId_to_movieName={}
    movieName_to_movieId={}
    
    def loadMovieLensDataset(self):
       
        ratingsDataset = 0
        self.movieId_to_movieName={}
        self.movieName_to_movieId={}
        df1 = pd.read_csv(self.rating_file_location,skiprows=1)
        df1.columns=['user','item', 'rating', 'timestamp']
        df2 = pd.read_csv(self.my_rating_file_location,skiprows=1)
        df2.columns=['user','item', 'rating', 'timestamp']
        frame =[df1,df2]
        ratingsData = pd.concat(frame,ignore_index=True)
        print(ratingsData.head())
        ratingsData.to_csv( "ratings-Data.csv", index=False)

        ratingsData = ratingsData.astype({'user': str, 'item': str,'rating':str,'timestamp':str})
        print(ratingsData.tail())
        reader = Reader(line_format='user item rating timestamp')
        ratingsDataset = Dataset.load_from_df(ratingsData[['user', 'item', 'rating']], reader)
                
        #Open the movies file, use encoding='ISO-8859-1' to avoid Unicode decode error
        #If csvfile is a file object, it should be opened with newline=''.
        with open(self.movies_file_location, newline='',encoding='ISO-8859-1') as csv_file:
            #Return a reader object which will iterate over lines in the given csvfile.
            movie_Reader = csv.reader(csv_file)
            next(movie_Reader) #Skip the first/header line
            for row in movie_Reader:
                movieID = int(row[0])
                movieName = row[1]
                self.movieId_to_movieName[movieID] = movieName
                self.movieName_to_movieId[movieName] = movieID
                
        return ratingsDataset
    
    def getMovieName(self, movieID):
        if movieID in self.movieId_to_movieName:
            return self.movieId_to_movieName[movieID]
        else:
            return ""
        
    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.new_rating_csv,newline='') as cvsfile:
            ratingReader = csv.reader(cvsfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
        rank = 1
        for movieID, _ in sorted(ratings.items(), key=lambda x:x[1], reverse = True):
            rankings[movieID] = rank
            rank += 1
        return rankings
                
    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.movies_file_location, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line
            for row in movieReader:
                movieID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[movieID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield            
        
        return genres

    def getYears(self):
        expToMatch = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.movies_file_location, newline = '', encoding = 'ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)
            for row in movieReader:
                movieId = int(row[0])
                title =row[1]
                rawYear = expToMatch.search(title)
                year = rawYear.group(1)
                if year:
                    years[movieId]=int(year)
        return years
                
        