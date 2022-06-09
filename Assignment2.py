#!/usr/bin/env python
# coding: utf-8

import math
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')

df = pd.read_csv('dataset.csv', sep='\t', engine = 'python', encoding='WINDOWS-1252')


# Pre-Processing(Filtering, removing Stopwords)

df = df.replace('[-",?!.\n()]', '', regex=True)
df = df.replace('é', 'e', regex=True)
df['Review'] = df['Review'].str.lower()
df['Review'] = df['Review'].str.strip()
df['Review'] = df['Review'].apply(lambda words: ' '.join([word for word in words.split() if word not in stop]))
total_reviews = df.size
words_dict = {}

# Tokenization 
# if word exists count++
# else add word and count=1
for item in range(total_reviews):
    review = df.iloc[item]['Review']
    tokenList = word_tokenize(''.join(review))        
    for token in tokenList:
        if token in words_dict.keys():
            words_dict.update({token : words_dict[token] + 1})
        else:
            words_dict.update({token : 1})

# size of dictionary
len(words_dict)

# dividing each word to get the probabilty of each word
probs = {}
for key, value in words_dict.items():
     probs.update({key : value / total_reviews}) 


nested_dict = {}

# Tokens comparison
# update count if word exists otherwise set value to 1
for reviewx in range(total_reviews):
    index = 0
    nextWord = 1
    word = 0
    review = df.iloc[reviewx]['Review']
    tokenList = word_tokenize(''.join(review))
    
    for tokenx in tokenList:
        if(word < len(tokenList)):
            probA = tokenList[word]
            
            for i in range(len(tokenList)):
                if(nextWord + 1 < len(tokenList)):
                    probB = tokenList[nextWord]
                    if probA in nested_dict.keys():
                        if probB in nested_dict[tokenx].keys():
                            nested_dict[probA][probB] = nested_dict[probA][probB] + 1
                        else:
                            nested_dict[probA][probB] = 1
                    else:
                          nested_dict[probA] = {probB : 1}

                    nextWord = nextWord + 1
                    index = index + 1
            word = word + 1
            nextWord = word + 1
            index = 0

len(nested_dict)

scores = pd.DataFrame({}, columns = ['A', 'B', 'MI Score'])



# finding the entropy and conditional entropy
# Mutual information score using the difference of two
for word in nested_dict.keys(): 
    for nextWord in nested_dict[word]:
        probA = probs[word]
        countA = words_dict[word]
        countB = words_dict[nextWord]
        countAwithBINT = nested_dict[word][nextWord]
        countAwithB = float(countAwithBINT)
        probAwithB = countAwithB / countB
        
        if(probA < probAwithB):
            try:
                entropyA = - 1 * probA * (math.log(probA, 2) + 0.1)
                entropyAtoB = -1 * probAwithB * (math.log(probAwithB, 2))  - (1 - probAwithB) * (math.log(1 - probAwithB, 2))
                mi = -(entropyA - entropyAtoB) 
                scores = scores.append({'A': word, 'B':nextWord, 'MI Score': mi}, ignore_index = True)
            except ValueError:
                continue
# Results
scores = scores.sort_values(by = 'MI Score', ascending = False)
scores = scores.reset_index(drop = True)
scores = scores