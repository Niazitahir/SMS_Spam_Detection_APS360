import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import pandas as pd
import csv
from sklearn.utils import shuffle
import nltk
#nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#python3 -m pip install autocorrect
from autocorrect import Speller
spell = Speller()
'''NOTES:
THIS SCRIPT RANDOMELY SPLITS THE DATASET INTO 70% TRAINING,
15% Valid, 15% Test

Change the values as needed to change training params

It is then saved into the RawData Folder as a csv. Simply
Import the csv using pandas from_csv() function to recreate
the dataframes. 

Have fun, boys. 
'''


#remove ham examples to increase spam percentage or increase spam
#fuck with spam examples to mess (i.e grayscale, scaling)


def textPrep():
    stopW = set(stopwords.words('english'))
    totalData = []
    with open('RawData/sms+spam+collection/SMSSpamCollection', newline = '') as csvfile:
        spamreader = csv.reader(csvfile, quotechar='|')
        for row in spamreader:
            totalData.append(row[0])
            
    labels = []
    values = []
    for row in totalData:
        try:
            row.split()[1]
            labels.append(row.split()[0]) 
            #ans = " ".join(row.split()[1:]) #lower casing text and lemmatize
            ans = row.split()[1:]
            #word_tokens = word_tokenize(ans)
            filtered_ans = [w.lower() for w in ans if not w.lower() in stopW]
            filtered_ans = spell(" ".join(filtered_ans))
            values.append(filtered_ans)
        except:
            print("opp got em chief") #do not add data without a label/value match. some only have a label
            
    print(len(values), len(labels))
    totalData = list(zip(labels, values))
    print(totalData[:10])

    totalData = pd.DataFrame(totalData)

    df = shuffle(totalData)

    print(totalData.head)
    testData= df.iloc[int(len(df)*0.85):, :] #15%
    validData = df.iloc[int(len(df)*0.7):int(len(df)*0.85), :]#15%
    trainData = df.iloc[:int(len(df)*0.7), :]#70%

    print(testData.shape, validData.shape, trainData.shape)
    testData.to_csv("RawData/testData", header = None)
    validData.to_csv("RawData/validData", header = None)
    trainData.to_csv("RawData/trainData", header = None)

def w2VecPreparation():
    import gensim
    import logging
    sentences = gensim.models.word2vec.LineSentence(open('RawData/trainData',encoding="utf8"), max_sentence_length=10000)
    model = gensim.models.Word2Vec(sentences, window=15, min_count=3, workers=4) #https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-sen.2018.5046
    model.save("RawData/train.w2v")
    # Get the ordered list of words in the vocabulary
    words = list(w for w in model.wv.index_to_key)
    # Make a dictionary
    we_dict = {word:model.wv[word] for word in words}
    print(we_dict)
def adaPrep():
    from transformers import GPT2TokenizerFast
    totalData = []
    tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/text-embedding-ada-002')
    assert tokenizer.encode('hello world') == [15339, 1917]
    
    with open("RawData/trainData", encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, quotechar='|')
        for row in spamreader:
            totalData.append(row)

    values = []
    labels  = []
    for i in range(len(totalData)):
        labels.append( tokenizer.encode(totalData[i][1]))
        values.append(tokenizer.encode(totalData[i][2]))
    totalData = list(zip(labels, values))
    totalData = pd.DataFrame(totalData).to_csv("RawData/trainEncoded", header=False)
    totalData = []
    with open("RawData/testData", encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, quotechar='|')
        for row in spamreader:
            totalData.append(row)

    values = []
    labels  = []
    for i in range(len(totalData)):
        labels.append( tokenizer.encode(totalData[i][1]))
        values.append(tokenizer.encode(totalData[i][2]))
    totalData = list(zip(labels, values))
    totalData = pd.DataFrame(totalData).to_csv("RawData/testEncoded", header=False)
    totalData = []
    with open("RawData/validData", encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, quotechar='|')
        for row in spamreader:
            totalData.append(row)

    values = []
    labels  = []
    for i in range(len(totalData)):
        labels.append( tokenizer.encode(totalData[i][1]))
        values.append(tokenizer.encode(totalData[i][2]))
    totalData = list(zip(labels, values))
    totalData = pd.DataFrame(totalData).to_csv("RawData/validEncoded", header=False)
        
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
if __name__ == "__main__":
    #textPrep()
    adaPrep()
    
#wor2vec implementation, save  vectors to file 

