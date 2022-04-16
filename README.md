# cs5293sp22-project2  

# Cuisine Predictor    

## Author: PAVAN VASANTH KOMMMINENI   

## Project Summary: Created an  applications that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals.The yummly data is accessed from the link attached in project description.("https://www.dropbox.com/s/f0tduqyvgfuin3l/yummly.json")

To run the project change the directory to project2 and execute the command -  
pipenv run python project2.py --N 5 --ingredient "eggs"  --ingredient "romaine lettuce"   --ingredient garlic --ingredient "salt"  

project2.py: In this file all functions and modules are defined.  

Python Functions:  
1.Input data function  
2.Filtered data function  
3.Vectorization function  
4.Knn Classification function  
5.Knn Predictor function  
6.Display function  

The cd is cs5293sp22-Project2; I'm listing project2 for the rootpath.  

Required Python Packages:  

1.Pandas:For the Dataframe creation.  
2.Json: For the importing of data and to get the required output format.  
3.nltk: For the stop words filtering and flattening the lists.  
4.Counter: to get count of elements in the list  
5.Sklearn  
  KneighborsClassifier: For the knn prediction algorithm.  
  metrics.pairwise: for the cosine similarity. 
6.argparse: For the arguments passing.  

Functions:  

Input data function:(input_data()):  
1.This function extracts the data from data.json file.  
2.In this I'm converting it to dataFrame and removing the duplicate id's.  
3.I'm returning the filtered data.  

Filtered data function(filtered_words()):  
1.In this function; I'm making a list of all ingredients with no dupliacates.  
2.I'm taking input as data returned from input-data() function   
3.I'm returning the filtered_words (all ingredients)  

Vectorize function(vectorize()):  
1.This function takes the input data and ingredient data.  
2.This function is kind of vectorization, which will return the count of each ingredient(in this case either 1 or 0)  
3.It returns the count dataframe and count list data.  

Knn Classification function(knnClassifier):  
1.This function builds the training model using knn neighbor algorithm.  
2.It takes the count dataframe and input data as input.  
3.It returns the knn model which generated from the data.  

Knn Predictor(knnPredictor):  
1.This function predicts the given input and gives the nearest neighbors.  
2.In this function i'm generating a count list similar to the one generated in vectorization function and converted to dataframe.  
3.Using the prediction algorithm; returning the score,cuisine,input count list and matchids.  

Display Function(Display()):  
1.This Function is used to print the output in required format.  
2.Using Cosine similarity module I'm finding similarity btw the input count list and match_id list  
3.I'm returning zero for the test case.  

Main Function  
1.The execution of project starts.  
2.All input arguments are parsed in this file.The arguments below are parsed  
--N --ingredient  

Test Cases:  
test_input_data:  
This method is used to test input_files functionality.It asserts true when the len(data)!=0.  

test_filtered_data:  
This method is used to test filtered words functionality.It asserts true when len of filtered!=0  

test_vectorize:  
This method is used to test vectorize functionality. It asserts true when len of vector!=0. 

test-knnclassifier:  
This method is used to test knnclassifier functionality. It asserts true when the function is calling.  

test_knnPredictor:  
This method is used to test knnPredictor functionality. It asserts True when w!=0  

test_Display:  
This method is used to test Display functionality. It asserts True when the returned value to zero.  

Assumptions:  
In the KnnClassifier, I'm using the default neighbors (n=5). 

Bugs:  
Some times the closest id isn't coming in descending order.  
For Contrast combinations The closest Id not coming.  

References:  
https://stackoverflow.com/questions/58460304/vectorize-a-list-of-words-by-a-vector-of-frequencies-of-the-words   
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  
https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists  

I increased the instance to 8gb in gcp to execute.  

