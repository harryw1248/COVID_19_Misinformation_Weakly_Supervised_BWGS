import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from tqdm import tqdm
import torch
import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functools import partial
import pickle
from collections import deque
stop_words = set(stopwords.words('english')) 
import time
import marshal
import glob
import time


class BWGSModel(object):
    
    def __init__(self, df, model, tokenizer, graph, outputFolder, combinedOutputFolder, modelOutputFolder = './', queue=None, useMasterEmb = False, masterContrib = 0.5, similarityThreshold = 0.3, embeddingType='last4sum', numThreshold= 10000, saveEveryDepth = False, numComp = 10000):
        
        self.df = df
        self.model = model
        self.tokenizer = tokenizer
        self.graph = graph
        self.outputFolder = outputFolder
        self.combinedOutputFolder = combinedOutputFolder
        self.embeddingType = embeddingType
        self.numThreshold = numThreshold
        self.similarityThreshold = similarityThreshold
        self.saveEveryDepth = saveEveryDepth
        self.modelOutputFolder = modelOutputFolder
        self.numComp = numComp
        self.words_explored = []
        
        if queue is None:
            self.q = deque()
        else:
            self.q = queue
            
            
        self.masterEmb = None
        
        self.useMasterEmb = useMasterEmb
        self.masterContrib = masterContrib
        
        self.masterEmbList = []
        
        self.generateStates()
        
        
    def generateStates(self):
        
        
        for i in tqdm(range(len(self.df))):
            
            if os.path.exists(os.path.join(self.outputFolder, f"{i}.msh")):
                continue


            tokens = self.tokenizer.encode(self.df.iloc[i]['message'].lower())
            decoded = self.tokenizer.decode(tokens).split(" ")
            logits, hidden_states = self.model(torch.Tensor(tokens).unsqueeze(0).long())

            hidden_states = torch.stack(hidden_states).squeeze(1).permute(1,0,2)

            
            if self.embeddingType == 'last4sum':
                embedding = torch.sum(hidden_states[:,9:13,:],1)
            elif self.embeddingType =='last4concat':
                embedding = hidden_states[tokenIndex,9:13,:].reshape(-1)
            elif self.embeddingType == 'secondlast':
                embedding = hidden_states[tokenIndex,-2,:]
            else:
                embedding = hidden_states[tokenIndex,-1,:]
                    
                    
            embedding = embedding.detach().cpu().numpy()
            
            marshal.dump(embedding.tolist(), open(os.path.join(self.outputFolder, f"{i}.msh"), 'wb'))
        
        
        
        
    def getSymptomEmbedding(self, symptom, subset = None):
    
        embeddingList = []
        messageList = []

        symptomToken = self.tokenizer.convert_tokens_to_ids(symptom)

        for i in range(len(self.df)):

            if symptomToken in self.tokenizer.encode(self.df.iloc[i]['message'].lower()):

                tokens = self.tokenizer.encode(self.df.iloc[i]['message'].lower())
                decoded = self.tokenizer.decode(tokens).split(" ")

                hidden_states = np.array(marshal.load( open(os.path.join(self.outputFolder, f"{i}.msh"), 'rb') ))

                try:
                    tokenIndex = tokens.index(symptomToken)
                except:
                    a= 1
                    continue

 
                embedding = hidden_states[tokenIndex,:]

                embeddingList.append(embedding)
                messageList.append(self.df.iloc[i]['message'].lower())

                if len(embeddingList)==30:
                    break



        return embeddingList, messageList
    
    
    
    
    def getSimilarWords(self, symptom, embList):
    
     
        output = []


        symptomToken = self.tokenizer.encode(symptom)[1]

        for i in tqdm(range(len(self.df))):
        
            tokens = self.tokenizer.encode(self.df.iloc[i]['message'].lower())

            if symptomToken in tokens:

                
                hidden_states = np.array(marshal.load( open(os.path.join(self.outputFolder, f"{i}.msh"), 'rb') ))

                similarity = cosine_similarity(hidden_states, embList.reshape(1,-1)).reshape(-1)


                index = np.where([similarity> self.similarityThreshold])[1]
                
                try:
                    selectTokens = np.array(tokens)[index]
                except:
                    print(i)
                    print(index)
                    print(hidden_states.shape)
                    print(len(tokens))
                    print(len(self.tokenizer.encode(self.df.iloc[i]['message'].lower())))
                    print(self.df.iloc[i]['message'])
                    break
                    
                selectSim = similarity[index]


                for j in range(len(index)):
                    token = self.tokenizer.ids_to_tokens[selectTokens[j]]
                    sim = selectSim[j]
                    output.append((token, sim,i))


            if i==self.numThreshold:
                break

        return output
        
    
    
    def getOutput(self, out):
    
        output = out

        outMap = {}

        for i in range(len(output)):
            if output[i][0] in outMap:
                outMap[output[i][0]].append(output[i][1])
            else:
                outMap[output[i][0]] = [output[i][1]]


        outMap_ = {}

        for i in range(len(output)):
            if output[i][0] in outMap_:
                outMap_[output[i][0]].append(output[i][2])
            else:
                outMap_[output[i][0]] = [output[i][2]]


        outputDf = []

        for key in outMap.keys():
            length = len(outMap[key])
            mean = np.mean(outMap[key])

            outputDf.append([key, length, mean])

        outputDf = pd.DataFrame(outputDf)
        outputDf.columns = ['word','counts','mean_sim']
        outputDf = outputDf.sort_values('mean_sim', ascending=False)

        return outputDf, outMap, outMap_
    
    
    
    
    def exploreNode(self, word, depth, maxDepth = 3, topk = 5):

    
        self.graph.addNode(word,0,depth)

        print(f"Depth : {depth} Exploring {word}")

        if depth == maxDepth:
            print("Reached max depth")
            return

        keyWord = word

        token = self.tokenizer.encode(keyWord)[1]

        if self.graph[word].vector is None:

            inEdgeList = self.graph[word].edges_in

            if len(inEdgeList)==0:
                textIDList = None
            else:
                textIDList = []

                for edge in inEdgeList:
                    textIDList.append(self.graph.edgeList[edge].textID)

                textIDList = list(set(list(itertools.chain.from_iterable(textIDList))))

            
            embList,msgList = self.getSymptomEmbedding(keyWord, subset = textIDList)

            meanEmb = np.array(embList)
            meanEmb = np.mean(meanEmb,0)


            self.graph[word].vector = meanEmb
            
            if self.masterEmb is None:
                self.masterEmb = meanEmb
            
            dist = getCosineDist(meanEmb, self.masterEmb)
            
            self.graph[word].masterDist = dist

        else:
            meanEmb = self.graph[word].vector
            
            if self.masterEmb is None:
                self.masterEmb = meanEmb
                
            dist = getCosineDist(meanEmb, self.masterEmb)
            
            self.graph[word].masterDist = dist


        symptom_ =''
        embList_ = meanEmb

        if self.useMasterEmb:
            
            finalEmb = self.masterContrib*self.masterEmb + (1 - self.masterContrib)*meanEmb
            
            out = self.getSimilarWords( symptom_, finalEmb)
        else:
            out = self.getSimilarWords( symptom_, meanEmb)

        outputDf, outMap, outMap_ = self.getOutput(out)

        outputDf = outputDf[outputDf.word!=keyWord]
        outputDf = outputDf.sort_values('mean_sim', ascending=False)
        outputDf = outputDf.head(topk)

        outputDf = outputDf[outputDf.mean_sim>0.4]

        print(outputDf)
        print("-----------------------")

        for i in range(len(outputDf)):

            word = outputDf.iloc[i]['word']
            numCount = outputDf.iloc[i]['counts']
            weight = outputDf.iloc[i]['mean_sim']
            textIDs = outMap_[word]

            wordList = set(self.graph.wordMap.keys())

            self.graph.addNode(word,0,depth+1)
            self.graph[word].textIDList.append(textIDs)
            self.graph.addEdge(keyWord, word, numCount, weight, textIDs)

            if word in wordList:
                continue

            self.words_explored.append(word)
            self.q.append((word, depth+1))
            
            
    def trainModel(self, maxDepth = 3, topk = 5):
        
        currDepth = 0
        
        while len(self.q)>0:
            token, depth = self.q.popleft()
            
            if depth> currDepth:
                
                if self.saveEveryDepth:
                    filepath = os.path.join( self.modelOutputFolder, f"depth_{currDepth}.pkl")
                    self.saveModel(filepath)
                
                self.masterEmbList.append(self.masterEmb.copy())
                self.getMeanEmbedding(depth-1)
                currDepth += 1
            
            self.exploreNode(word = token, depth = depth, maxDepth=maxDepth, topk=topk)

        filepath = os.path.join(self.modelOutputFolder, "final.pkl")
        self.saveModel(filepath)

    def getMeanEmbedding(self, depth, topk = 3):
        
        candidates = self.graph.depthMap[depth]
        vals = [self.graph[x].masterDist for x in candidates]
        vals = [(x,y) for x,y in zip(candidates,vals)]
        vals = sorted(vals, key = lambda x : -x[1])
        meanEmb = self.masterEmb
        
        for i in range(min(topk, len(vals)) ):
            meanEmb += self.graph[ vals[i][0] ].vector
            
        meanEmb = meanEmb/(topk+1)
        self.masterEmb = meanEmb
        
        print("Master Embedding updated")
        
        
    
    def plotGraph(self):
        
        edgeList, nodeList, nodeValues, nodeCount, nodeText, nodeSize = getGraphComponents(self.graph)
        G=nx.Graph()
        G.add_nodes_from(nodeList)
        G.add_edges_from(edgeList)
        edge_trace, node_trace1, node_trace = getPlotlyComponents(G, nodeList, nodeSize, nodeValues, nodeText)


        fig = go.Figure(data=[edge_trace, node_trace1, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=50),

                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        
        fig.show()
        
        
    def saveModel(self,filename):
        
        classDict = self.__dict__.copy()
        classDict.pop('model')
        classDict.pop('tokenizer')
        classDict.pop('df')
        
        pickle.dump( classDict, open( filename, "wb" ) )
        
        
    def loadModel(self, filename):
        
        classDict = pickle.load(open(filename, 'rb'))
        
        for key in list(classDict.keys()):
            self.__dict__[key] = classDict[key]
        
        
    def get_words_explored(self):
        return self.words_explored

    
def clear_model_cache(outputFolder, combinedOutputFolder, modelFolder):
    files = glob.glob((outputFolder+'/*'))
    for f in files:
        os.remove(f)

    files = glob.glob((combinedOutputFolder+'/*'))
    for f in files:
        os.remove(f)

    files = glob.glob((modelFolder+'/*'))
    for f in files:
        os.remove(f)

    print("Model cache cleared")
    

def run_model(df, bert_model, tokenizer, relevant_tweets_csv_name, outputFolder, combinedOutputFolder, modelFolder, resultsFolder, df_train=None, create_df_train=True,seed_word='myth', similarityThreshold=0.4, maxDepth=1, topk=4, num_sample_tweets=100):
    
    startTime = time.time()
    clear_model_cache(outputFolder, combinedOutputFolder, modelFolder)
    
    if create_df_train:
        df_train = df[df.message.str.contains(seed_word,case=False)]
        df_train = df_train.head(num_sample_tweets)
    
    graph = nx.Graph()

    q = deque()
    q.append((seed_word,0))

    q = deque()
    q.append((seed_word,0))
    BWGS = BWGSModel(df_train, bert_model, tokenizer, graph, outputFolder, combinedOutputFolder, modelOutputFolder = modelFolder, 
               queue = q,  useMasterEmb=True, masterContrib=0.3, similarityThreshold=similarityThreshold, numThreshold=10000)

    print("Training Model")
    BWGS.trainModel(maxDepth, topk)
    words_explored = BWGS.get_words_explored()
    words_explored.append(seed_word)
    print(words_explored)

    clear_model_cache(outputFolder, combinedOutputFolder, modelFolder)
    
    print(time.time() - startTime)
    
    return words_explored

'''
BMDWGS
'''
def run_model_multi_directional(df, bert_model, tokenizer, relevant_tweets_csv_name, outputFolder, combinedOutputFolder, modelFolder, resultsFolder, df_train=None, create_df_train=True, seed_word_1='myth', seed_word_2='false', similarityThreshold=0.4, maxDepth=1, topk=4, num_sample_tweets=50, and_option=True):
    
    startTime = time.time()
    clear_model_cache(outputFolder, combinedOutputFolder, modelFolder)
    
    if create_df_train:
        if and_option:
            df_train = df[df.message.str.contains(seed_word_1,case=False) & df.message.str.contains(seed_word_2,case=False)]
            df_train = df_train.head(num_sample_tweets)
        else:
            df_train = df[df.message.str.contains(seed_word_1,case=False) | df.message.str.contains(seed_word_2,case=False)]
            df_train = df_train.head(num_sample_tweets)
            
    
    #first seed word
    graph1 = Graph()

    q1 = deque()
    q1.append((seed_word_1,0))

    BWGS1 = BWGSModel(df_train, bert_model, tokenizer, graph1, outputFolder, combinedOutputFolder, modelOutputFolder = modelFolder, queue = q1,  useMasterEmb=True, masterContrib=0.3, similarityThreshold=similarityThreshold, numThreshold=10000)

    print("Training Model 1")
    BWGS1.trainModel(maxDepth, topk)
    words_explored_1 = BWGS1.get_words_explored()
    words_explored_1.append(seed_word_1)
    print("words_explored_1")
    print(words_explored_1)
    
    clear_model_cache(outputFolder, combinedOutputFolder, modelFolder)
    
    #second seed word
    graph2 = Graph()

    q2 = deque()
    q2.append((seed_word_2,0))

    BWGS2 = BWGSModel(df_train, bert_model, tokenizer, graph2, outputFolder, combinedOutputFolder, modelOutputFolder = modelFolder, queue = q2,  useMasterEmb=True, masterContrib=0.3, similarityThreshold=similarityThreshold, numThreshold=10000)

    print("Training Model 2")
    BWGS2.trainModel(maxDepth, topk)
    words_explored_2 = BWGS2.get_words_explored()
    words_explored_2.append(seed_word_2)
    
    print("words_explored_2")
    print(words_explored_2)
    
    print("Final word list")
    words_explored = words_explored_1 + words_explored_2
    print(words_explored)
        

    clear_model_cache(outputFolder, combinedOutputFolder, modelFolder)
    
    print(time.time() - startTime)
    
    print(words_explored)
    return words_explored

def get_relevant_tweets(df, words_explored, resultsFolder, relevant_tweets_csv_name):
    
    startTime = time.time()
    
    print("Finding relevant tweets")
    relevant_tweets_df = df[df.message.str.contains('|'.join(words_explored), case=False)]['message']    
    relevant_tweets_df.to_csv(resultsFolder + relevant_tweets_csv_name)
    relevant_tweets_df = pd.read_csv(resultsFolder + relevant_tweets_csv_name)
    relevant_tweets_df = relevant_tweets_df.drop_duplicates(subset='message', keep='last')
    relevant_tweets_df = relevant_tweets_df['message']
    relevant_tweets_df.to_csv(resultsFolder + relevant_tweets_csv_name)
    
    num_tweets = relevant_tweets_df.shape[0]
    print("num_tweets= " + str(num_tweets))

    clear_model_cache(outputFolder, combinedOutputFolder, modelFolder)
    
    print(time.time() - startTime)
    return num_tweets

#get tweets not classified as misinformation
def get_relevant_tweets_plus(df, words_explored, resultsFolder, outputFolder, combinedOutputFolder, modelFolder, relevant_tweets_csv_name):
    

    
    print("Finding relevant tweets")
    relevant_tweets_df = df[df.message.str.contains('|'.join(words_explored), case=False)]['message']    
    relevant_tweets_df.to_csv(resultsFolder + relevant_tweets_csv_name)
    relevant_tweets_df = pd.read_csv(resultsFolder + relevant_tweets_csv_name)
    relevant_tweets_df = relevant_tweets_df.drop_duplicates(subset='message', keep='last')
    relevant_tweets_df = relevant_tweets_df['message']
    relevant_tweets_df.to_csv(resultsFolder + relevant_tweets_csv_name)
    
    num_tweets = relevant_tweets_df.shape[0]
    print("num_tweets= " + str(num_tweets))
    
    print("checkpoint")
    print("Finding counter-relevant tweets")
    df1 = df
    df2 = relevant_tweets_df
    common = df1.merge(df2,on=['message'])
    df3 = df1[(~df1.message.isin(common.message))&(~df1.message.isin(common.message))]
    df3 = df3.head(num_tweets)
    df3.to_csv(resultsFolder + relevant_tweets_csv_name + "_counter_tweets")
    
    
    clear_model_cache(outputFolder, combinedOutputFolder, modelFolder)
    return num_tweets

def get_labeled_tweets(df, words_explored, resultsFolder, outputFolder, combinedOutputFolder, modelFolder, relevant_tweets_csv_name):
    
    startTime = time.time()
    positive_labels = ["conspiracy","calling out or correction"]

    
    positives_df = df[df.annotation1.str.contains('|'.join(positive_labels), case=False) | df.annotation2.str.contains('|'.join(positive_labels), case=False)]#['full_text']  

    
    num_positives_total = positives_df.shape[0]
    print("num_positives_total: " + str(num_positives_total))
    
    retrieved_df = df[df.full_text.str.contains('|'.join(words_explored), case=False, na=False)]
    
    
    num_retrieved = retrieved_df.shape[0]
    print("num_retrieved: " + str(num_retrieved))
    
    true_positives_df = retrieved_df[df.annotation1.str.contains('|'.join(positive_labels), case=False) | df.annotation2.str.contains('|'.join(positive_labels), case=False, na=False)]
    num_true_positives = true_positives_df.shape[0]
    print("num_true_positives: " + str(num_true_positives))
    
    
    
    false_negatives_df = df[df.annotation1.str.contains('|'.join(positive_labels), case=False, na=False) | df.annotation2.str.contains('|'.join(positive_labels), case=False, na=False) & ~(df.full_text.isin(retrieved_df.full_text))]
    false_negatives_df.to_csv(resultsFolder + "conspiracy_CMU_false_negatives_labels.csv")
    
    print("false_negatives annotation1 being conspiracy")
    print(false_negatives_df[false_negatives_df.annotation1.str.contains("conspiracy", na=False)].shape)
    
    print("false_negatives annotation2 being conspiracy")
    print(false_negatives_df[false_negatives_df.annotation2.str.contains("conspiracy", na=False)].shape)
    
    print("false_negatives annotation1 being calling out or correction")
    print(false_negatives_df[false_negatives_df.annotation1.str.contains("calling out or correction", na=False)].shape)
    
    print("false_negatives annotation2 being calling out or correction")
    print(false_negatives_df[false_negatives_df.annotation2.str.contains("calling out or correction", na=False)].shape)
    
    
    false_negatives_df.rename(columns={"full_text": "message"},inplace=True)
    false_negatives_df["message"].to_csv(resultsFolder + "conspiracy_CMU_false_negatives_message.csv")
    
    print(retrieved_df.head(10))
    false_positives_df = retrieved_df[~(retrieved_df.annotation1.str.contains('|'.join(positive_labels), case=False, na=False)) & ~(retrieved_df.annotation2.str.contains('|'.join(positive_labels), case=False, na=False)) &(retrieved_df.full_text.isin(retrieved_df.full_text))]
    false_positives_df.to_csv(resultsFolder + "conspiracy_CMU_false_positives_labels.csv")
    
    false_positives_df.rename(columns={"full_text": "message"},inplace=True)
    false_positives_df["message"].to_csv(resultsFolder + "conspiracy_CMU_false_positives_message.csv")
    
    num_false_positives = false_positives_df.shape[0]
    num_false_negatives = false_negatives_df.shape[0]
    
    precision = num_true_positives / (num_true_positives + num_false_positives)
    recall = num_true_positives / (num_true_positives + num_false_negatives)
    f1_score = 2*(recall*precision) / (recall+precision)
    
    
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1_score: " + str(f1_score))
    print("latex format")
    print("%.4f & %.4f & %.4f" %(precision, recall, f1_score))
    
    retrieved_df.rename(columns={"full_text": "message"},inplace=True)
    retrieved_df = retrieved_df["message"]
    print("columns modified")
    retrieved_df.to_csv(resultsFolder + relevant_tweets_csv_name)
    

    clear_model_cache(outputFolder, combinedOutputFolder, modelFolder)
    
    print(time.time() - startTime)

def get_labeled_tweets_fn(df, words_explored, resultsFolder, outputFolder, combinedOutputFolder, modelFolder, relevant_tweets_csv_name):
    
 
    positive_labels = ["FALSE"]
    positives_df = df[df["Label"]==False]
    negatives_df = df[df["Label"]==True]

    num_positives_total = positives_df.shape[0]

    retrieved_df = df[df.text.str.contains('|'.join(words_explored), case=False, na=False)]         
    non_retrieved_df = df[~(df.text.isin(retrieved_df.text))]
    true_negatives_df = non_retrieved_df[non_retrieved_df["Label"]==True]
    num_true_negatives = true_negatives_df.shape[0]
    false_negatives_df = non_retrieved_df[non_retrieved_df["Label"]==False]
    num_false_negatives = false_negatives_df.shape[0]
    

    missed_df = df[df["Label"]==False & ~(df.text.isin(retrieved_df.text))]

    missed_df.to_csv("missed_covid_fn.csv", index=True)

    num_retrieved = retrieved_df.shape[0]
    true_positives_df = retrieved_df[retrieved_df["Label"]==False]
    num_true_positives = true_positives_df.shape[0]
    num_false_positives = num_retrieved - num_true_positives
    
    print("num_true_positives: " + str(num_true_positives))
    print("num_false_positives: " + str(num_false_positives))
    print("num_false_negatives: " + str(num_false_negatives))
    
    precision = num_true_positives / (num_true_positives + num_false_positives)
    recall = num_true_positives / (num_true_positives + num_false_negatives)
    f1_score = 2*(recall*precision) / (recall+precision)
    
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1_score: " + str(f1_score))
    print("latex format")
    print("%.4f & %.4f & %.4f" %(precision, recall, f1_score))

def cluster_into_super_labels(df, resultsFolder):
    misinformation_labels = ["conspiracy", "fake cure", "fake treatment", "false fact or prevention", "false public health response"]
    
    fact_labels = ["true treatment", "true prevention", "true public health response", "calling out or correction"]

    reaction_labels = ["politics", "commercial activity or promotion", "emergency", "news", "panic buying"]

    other_labels = ["irrelevant", "ambiguous or hard to classify","sarcasm or satire"]
    
    
    misinformation_labels_df = df[df.annotation1.str.contains('|'.join(misinformation_labels), case=False) | df.annotation2.str.contains('|'.join(misinformation_labels), case=False)]
    
    fact_labels_df = df[df.annotation1.str.contains('|'.join(fact_labels), case=False) | df.annotation2.str.contains('|'.join(fact_labels), case=False)]

    reaction_labels_df = df[df.annotation1.str.contains('|'.join(reaction_labels), case=False) | df.annotation2.str.contains('|'.join(reaction_labels), case=False)]
    
    other_labels_df = df[df.annotation1.str.contains('|'.join(other_labels), case=False) | df.annotation2.str.contains('|'.join(other_labels), case=False)]
    
    misinformation_labels_df.rename(columns={'full_text': 'message'},inplace=True)
    fact_labels_df.rename(columns={'full_text': 'message'},inplace=True)
    reaction_labels_df.rename(columns={'full_text': 'message'},inplace=True)
    other_labels_df.rename(columns={'full_text': 'message'},inplace=True)
    
    misinformation_labels_df = misinformation_labels_df['message']
    fact_labels_df = fact_labels_df['message']
    reaction_labels_df = reaction_labels_df['message']
    other_labels_df = other_labels_df['message']
    
    print("num misinformation: " + str(misinformation_labels_df.shape[0]))
    print("num fact: " + str(fact_labels_df.shape[0]))
    print("num reaction: " + str(reaction_labels_df.shape[0]))
    print("num other: " + str(other_labels_df.shape[0]))
    
    misinformation_labels_df.to_csv(resultsFolder + "misinformation_labels")
    fact_labels_df.to_csv(resultsFolder + "fact_labels")
    reaction_labels_df.to_csv(resultsFolder + "reaction_labels")
    other_labels_df.to_csv(resultsFolder + "other_labels")
    