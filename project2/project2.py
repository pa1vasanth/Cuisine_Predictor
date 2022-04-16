import pandas as pd
import json
import os
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import argparse
nltk.download('stopwords')
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
def input_data():
    data_file='data.json'
    f = open(data_file)
    data = json.load(f)
    df = pd.DataFrame(data)
    f.close()
    df=df.drop_duplicates(['id'])
    return df

def filtered_data(df):
    filtered=[]
    sdf=df['ingredients']
    for i in sdf:
        filtered.append(i)
    ingredients_data=nltk.flatten(filtered)
    unique_ing=[]
    for x in ingredients_data:
        if x not in unique_ing:
            unique_ing.append(x)
    filtered_words = [word for word in unique_ing if word not in stopwords.words('english')]
    
    return filtered_words

def vectorize(df,filtered_words):
    vector=[]
    sdf=df['ingredients']
    for j in sdf:
        c = Counter(j)
        w=[c[i] for i in filtered_words]
        vector.append(w)
    vect_df=pd.DataFrame(vector)
    vect_df.columns=filtered_words

    return vect_df,vector

def knnClassifier(data,vec_df):
    cuisine_df=data['cuisine']
    knn = KNeighborsClassifier()
    knn_class=knn.fit(vec_df,cuisine_df)
    return knn_class

def knnPredictor(knn,filtered,vector,input_):
    c = Counter(input_)
    w=[c[i] for i in filtered]
    vect=pd.DataFrame([w])
    cuisine = knn.predict_proba(vect)[0]
    single_cuisine = knn.predict(vect)
    classe = knn.classes_
    dist,m_id = knn.kneighbors(vect)
    return m_id,single_cuisine,cuisine,w

def Display(m_id,single_cuisine,cuisine,w,vector,data,N):
    id_df=data['id']
    id=[]
    for i in id_df:
        id.append(i)
    Closest_id=[]
    for i in range(len(m_id[0])):
        Closest_id.append(id[m_id[0][i]])
    
    closest_vect=[]
    q=list(m_id)
    for i in range(len(q[0])):
        closest_vect.append(vector[q[0][i]])
    eyu=[]
    for i in range (len(closest_vect)):
        o=closest_vect[i]
        id_score=cosine_similarity([w] , [o])[0][0]
        eyu.append("{:.2f}".format(id_score))
    Cl=[]
    for i in range (N):
        dict={'id':Closest_id[i],'score':float(eyu[i])}
        Cl.append(dict)
    dicti={"cuisine":nltk.flatten((single_cuisine.tolist()))[0],"score":(max(cuisine)),'closest':Cl}
    di=json.dumps(dicti,indent=4)
    print(di)
    return 0



if __name__ == "__main__":
    

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--N",type = int, required = True, help = "Closest Id count" )
    arg_parser.add_argument("--ingredient",type = str, required = True, help = "Ingredient", action = "append" )
    args = arg_parser.parse_args()
    i_data=[]
    if args.ingredient:
        i_data.append(args.ingredient)

    n=args.N
    ing=nltk.flatten(i_data)
    
    data=input_data()
    filtered=filtered_data(data)
    vector_df,vector=vectorize(data,filtered)
    knn=knnClassifier(data,vector_df)
    m_id,single_cuisine,cuisine,w=knnPredictor(knn,filtered,vector,ing)
    Display(m_id,single_cuisine,cuisine,w,vector,data,n)


