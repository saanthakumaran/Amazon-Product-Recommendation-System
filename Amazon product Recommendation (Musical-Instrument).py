# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 01:16:54 2020

@author: admin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip

#importing the dataset of musical instruments and customer rating 
ratings = pd.read_csv("ratings_Musical_Instruments.csv", header=None, index_col=False)
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#importing the meta data of the product description
df = getDF('meta_Musical_Instruments.json.gz')

ratings.columns=['UserId','ItemId','Ratings','Time']

ratings= ratings.iloc[:,0:3]

new=pd.DataFrame(ratings.groupby('ItemId')['Ratings'].mean())

new['number of ratings']=ratings.groupby('ItemId')['Ratings'].count()

import seaborn as sns

sns.set_style('white')
%matplotlib inline

plt.figure(figsize=(10,4))

new['number of ratings'].hist(bins =40)
  
new=new.reset_index()

new1=new[['ItemId','number of ratings']]

ratings= pd.merge(ratings, new1, on='ItemId')

ratings_new= ratings[ratings['number of ratings']>=100]

ratings_new=ratings_new.iloc[:,0:3]

from scipy.sparse import csr_matrix

from pandas.api.types import CategoricalDtype

user_u = CategoricalDtype(sorted(ratings_new.UserId.unique()), ordered=True)
item_u = CategoricalDtype(sorted(ratings_new.ItemId.unique()), ordered=True)

row = ratings_new.UserId.astype(user_u).cat.codes
col = ratings_new.ItemId.astype(item_u).cat.codes
sparse_matrix = csr_matrix((ratings_new["Ratings"], (row, col)), \
                           shape=(user_u.categories.size, item_u.categories.size))


data = ratings_new['Ratings'].tolist()


sparse_matrix.todense()    

#Creating a sparse dataframe with customer and products
solve = pd.SparseDataFrame(sparse_matrix, \
                         index=user_u.categories, \
                         columns=item_u.categories, \
                         default_fill_value=0)

#Cosider the choosen product is Yamaha PKBS1 Single Braced Adjustable X-Style Keyboard Stand
B000VSKPZG_similar = (solve['B000VSKPZG']) 

#Finding the product with similar rating based on correlation
similar_to_prod = solve.corrwith(B000VSKPZG_similar, axis=0)

corr_prod = pd.DataFrame(similar_to_prod, columns =['Correlation']) 
corr_prod.dropna(inplace = True) 

corr_prod.sort_values('Correlation', ascending = False).head(10) 
corr_prod=corr_prod.reset_index()  
corr_prod.head() 

corr_prod.columns=['ItemId','Correlation']

corr_prod= pd.merge(corr_prod, new1, on='ItemId')

corr_prod.head()

product_info = corr_prod

# The recommended musical products for the product Yamaha PKBS1 Single Braced Adjustable X-Style Keyboard Stand
product_info=pd.merge(product_info, df, left_on ="ItemId", right_on="asin", how='left')
sugg=product_info.sort_values('Correlation', ascending = False).head(11) 
