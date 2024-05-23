#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


movies_data=pd.read_csv(r'C:\Users\Hp\OneDrive\Documents\movies.csv')


# In[4]:


movies_data.head(10)


# In[5]:


# number of rows and columns in dataframe
movies_data.shape

#here row represents number of movies


# In[6]:


# selecting relevant features for recommendation

selected_features=['genres','keywords','tagline','cast','director']
print(selected_features)


# In[7]:


# replace null values with null string

for feature in selected_features:
    movies_data[feature]=movies_data[feature].fillna('')


# In[ ]:





# In[ ]:





# In[8]:


# combining all the 5 selected features

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']


# In[9]:


print(combined_features)


# In[10]:


# converting the text data to feature vectors

vectorizer= TfidfVectorizer()


# In[11]:


feature_vectors=vectorizer.fit_transform(combined_features)


# In[12]:


print(feature_vectors)


# In[14]:


# getting similarty scores using cosine similarity

similarity = cosine_similarity(feature_vectors)
print(similarity)


# In[15]:


print(similarity.shape)
#for each movie we will compare it with all the other movies


# In[43]:


#getting the movie name from the user

movie_name= input(' Enter your favrouite movie name : ')


# In[44]:


# creating a list with all the movies given in the dataset

list_of_titles=movies_data['original_title'].tolist()
print(list_of_titles)


# In[45]:


# finding the close match for the movie name given by the user

find_close_match=difflib.get_close_matches(movie_name,list_of_titles)
print(find_close_match)


# In[22]:


close_match=find_close_match[0]
print(close_match)


# In[24]:


# finding the index of the movie with title

index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
#The purpose of this line of code is to find the index of a movie in the movies_data DataFrame where the movie title matches the specified close_match. The resulting index_of_the_movie will be an integer representing the row index in the DataFrame where this movie is located.

#.values converts the Series to a numpy array.
print(index_of_the_movie)


# In[25]:


# getting list of similar movies
similarity_score=list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[26]:


len(similarity_score)


# In[27]:


#sorting the movies based on their similarity score

sorted_similar_movies=sorted(similarity_score, key=lambda x:x[1], reverse=True)
print(sorted_similar_movies)


# In[30]:


#print the name of similar movies based on their interest

print('movies suggested for you: \n')

i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if(i<30):
        print(i, '.', title_from_index)
        i+=1
    


# # compiling the whole code

# In[47]:


movie_name= input(' Enter your favrouite movie name : ')


list_of_titles=movies_data['original_title'].tolist()

find_close_match=difflib.get_close_matches(movie_name,list_of_titles)

close_match=find_close_match[0]

# finding the index of the movie with title

index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
#The purpose of this line of code is to find the index of a movie in the movies_data DataFrame where the movie title matches the specified close_match. The resulting index_of_the_movie will be an integer representing the row index in the DataFrame where this movie is located.

#.values converts the Series to a numpy array.

similarity_score=list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies=sorted(similarity_score, key=lambda x:x[1], reverse=True)

print('movies suggested for you: \n')

i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if(i<30):
        print(i, '.', title_from_index)
        i+=1
    


# In[ ]:





# In[ ]:





# In[ ]:




