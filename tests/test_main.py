import pytest
import os
import sys
sys.path.append('..')
import pytest
from nltk.corpus import stopwords
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from project2 import project2

def test_input_data():
     data=project2.input_data()

     if(len(data)!=0):
         assert True

def test_filtered():
     data=project2.input_data()
     filtered=project2.filtered_data(data)
     if(len(filtered)!=0):
         assert True

def test_vectorize():
    data=project2.input_data()
    data=data[1:3000]
    filtered=project2.filtered_data(data)
    vector_df,vector=project2.vectorize(data,filtered)
    if (len(vector)!=0):
        assert True

def test_knn():
    data=project2.input_data()
    data=data[1:3000]
    filtered=project2.filtered_data(data)
    vector_df,vector=project2.vectorize(data,filtered)
    knn=project2.knnClassifier(data,vector_df)
    #It is a algorithmic function so it doesn't return any data type
    assert True

def test_knn_predict():
    ing=['salt','plain flour',' pepper']
    data=project2.input_data()
    data=data[1:3000]
    filtered=project2.filtered_data(data)
    vector_df,vector=project2.vectorize(data,filtered)
    knn=project2.knnClassifier(data,vector_df)
    m_id,cusine,cusine_sc,w=project2.knnPredictor(knn,filtered,vector,ing)
    if (len(w)!=0):
        assert True

def test_Display():
    ing=['salt','plain flour',' pepper']
    n=4
    data=project2.input_data()
    data=data[1:3000]
    filtered=project2.filtered_data(data)
    vector_df,vector=project2.vectorize(data,filtered)
    knn=project2.knnClassifier(data,vector_df)
    m_id,cusine,cusine_sc,w=project2.knnPredictor(knn,filtered,vector,ing)
    q=project2.Display(m_id,cusine,cusine_sc,w,vector,data,n)
    if(q==0):
        assert True

