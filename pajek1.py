import networkx
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster.bicluster as cluster
from matplotlib import pyplot as plt
from sklearn.datasets import make_checkerboard
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn import datasets as ds
import wordcloud
import csv
import scipy.cluster.hierarchy as sch
import os
from sklearn import cluster as cl
from sklearn.metrics import consensus_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.cluster import KMeans


class PreProcessor():
    def __init__(self, filename):
        self.filename = filename
        self.graph = None
        self.edges = None
        self.nodeNames = None
        self.nodeList = None
        self.nodeIDdictionary = {}

    def readFile(self):
        self.graph = networkx.read_pajek(self.filename, encoding="UTF-8")

    def readEdges(self):
        self.edges = self.graph.edges()

    def getEdges(self):
        return self.edges

    def getEdgeLength(self):
        return len(self.edges)

    def getNodeNames(self):
        self.nodeNames = self.graph.nodes()

    def getNodeList(self):
        self.nodeList = self.graph.node

    def generateNodeIdDictionary(self):
        for i in self.nodeNames:
            self.nodeIDdictionary[i] = int(self.nodeList[i]["id"])

    def getNodeIDdictionary(self):
        return self.nodeIDdictionary

    def convertEdgesToIds(self, edgeList):
        idEdgeList = []
        for i in range(len(edgeList)):
            smallList = []
            for j in range(len(edgeList[i])):
                smallList.append(int(self.nodeList[edgeList[i][j]]["id"]))
            idEdgeList.append(smallList)
        return idEdgeList

    def cropeList(self,originalArray ,cropeLength):
        smallElement = []
        for i in range(cropeLength):
            smallElement.append(originalArray[i])

        return smallElement

    def breakClusterArrays(self, originalArray, clusterArray, n):
        brokenArray = []
        for i in range(n):
            brokenArray.append(originalArray[clusterArray == i])

        return brokenArray


class KmeansCluster():
    def __init__(self, dataSet, clusters):
        self.dataSet = dataSet
        self.clusters = clusters
        self.km = self.initializeKmeans(self.clusters)
        self.predicted = self.initializePrediction(self.km, self.dataSet)

    def initializeKmeans(self, n):
        return KMeans(n_clusters=n, init="k-means++", max_iter=300, n_init=10)

    def initializePrediction(self, km, dataset):
        return km.fit_predict(dataset)

    def getPrediction(self):
        return self.predicted

class SpectralClustering():
    def __init__(self, dataSet, clusters):
        self.dataSet = dataSet
        self.clusters = clusters
        self.sc = self.initializeSpectral(self.dataSet, self.clusters)

    def initializeSpectral(self, dataset, n):
        return cl.SpectralClustering(n_clusters=n).fit_predict(dataset)

    def getSpectral(self):
        return self.sc

class HeirarchialClustering():
    def __init__(self, dataSet, clusters):
        self.clusters = clusters
        self.dataSet = dataSet
        self.hc = self.initializeHeirarchical(self.dataSet, self.clusters)

    def initializeHeirarchical(self, dataSet, n):
        a = AgglomerativeClustering(n_clusters=n, affinity="euclidean", linkage="ward")
        b = a.fit_predict(dataSet)
        print(a.children_)
        plt.clf()
        plt.figure(figsize=(48, 48))
        sch.dendrogram(sch.linkage(a.children_))
        plt.title("Hierarchical Dendogram")
        plt.savefig(("Heirarchical" + "_" + "dendogram" + ".png"))
        # plt.clf()
        return b
        # sch.dendrogram(sch.linkage(self.dataset, method="ward"))
        # plt.title("Dendogram")
        # plt.ylabel("Euclidean Distance")


        # return AgglomerativeClustering(n_clusters=n, affinity="euclidean", linkage="ward").fit_predict(dataSet)

    def getHeirarchical(self):
        return self.hc

class dataFrameGenerator():
    def __init__(self, originalDataset, dataSet, nclusters, fileName, type):
        self.filename = fileName
        self.type = type
        self.dataSet = dataSet
        self.nclusters = nclusters
        self.originaldataset = originalDataset
        self.dic = self.initializeDictionary()
        self.maxvalue = self.getMAX()
        self.filler()
        self.df = self.initializeDataFrames()
        self.writeCSV()



    def writeCSV(self):
        filename = self.filename + "_" + self.type + "_" + str(self.nclusters)  + ".csv"
        self.df.to_csv(filename, index= False)
    def filler(self):
        for i in range(len(self.dic)):
            self.dic[i] = list(self.dic[i])
            if(len(self.dic[i]) < self.maxvalue):
                for j in range(self.maxvalue - len(self.dic[i])):
                    self.dic[i].append("nan")

    def getMAX(self):
        maxvalues = 0
        for i in range(len(self.dic)):
            if(maxvalues < len(self.dic[i])):
                maxvalues = len(self.dic[i])
        return maxvalues

    def initializeDictionary(self):
        dic = []
        for i in range(self.nclusters):
            dic.append(self.originaldataset[self.dataSet == i])
        return list(dic)


    def initializeDataFrames(self):
        df = pd.DataFrame()
        for i in range(self.nclusters):
            df["Cluster_" + str(i)] = self.dic[i]
        return df

    def getMax(self):
        maxNumber = 0
        print(self.dataSet)
        print(maxNumber)


class pltPlotter():
    def __init__(self, dataSet,clusteredset ,type, clusters, filename):
        self.dataset = dataSet
        self.type = type
        self.filename = filename
        self.clusteredSet = clusteredset
        self.nclusters = clusters
        if(type == "Heirarchical"):
            self.dendogram()
        self.Plot()

    def dendogram(self):
        plt.clf()
        plt.figure(figsize=(48,48))
        sch.dendrogram(sch.linkage(self.dataset, method="ward"))
        plt.title("Dendogram")
        plt.ylabel("Euclidean Distance")
        plt.savefig((self.filename+ "_" + "dendogram" + ".png"))

    def Plot(self):
        plt.clf()
        plt.figure(figsize=(48,48))
        listofColors = ["red", "navy", "blue", "mediumpurple", "yellow", "green", "cyan", "indigo"]
        for i in range(self.nclusters):
            plt.scatter(x = self.dataset[self.clusteredSet == i, 0], y = self.dataset[self.clusteredSet == i , 1], s=200, c = listofColors[i], label = ("Cluster_" + str(i)) )
        plt.legend()
        plt.title(self.type)
        plt.savefig((self.filename + "_" + self.type + ".png"))

class Controller():
    def __init__(self, fileName):
        self.fileName = fileName
        self.splitFileName = (os.path.splitext("index_interactions.csv")[0])
        self.originalFile = self.checker()
        self.sampleDataset = self.generateSampleDataset()
        self.startKmeans()
        self.startHeirarchical()
        self.startSpectral()
    def generateSampleDataset(self):
        test = PreProcessor(self.fileName)
        return np.array(test.cropeList(self.originalFile, 1000))


    def checker(self):
        if("csv" not in self.fileName):
            preprocessData = PreProcessor(self.fileName)
            preprocessData.readFile()
            preprocessData.readEdges()
            preprocessData.getNodeNames()
            preprocessData.getNodeList()
            preprocessData.generateNodeIdDictionary()

            edges = preprocessData.getEdges()
            originalFile = preprocessData.convertEdgesToIds(edges)
            return originalFile
        else:
            idedges = pd.read_csv(self.fileName, encoding="ISO-8859-1").convert_objects(convert_numeric=True)
            originalFile = idedges.iloc[:, :].values
            return originalFile


    def startKmeans(self):
        kMeanz = KmeansCluster(self.sampleDataset, 5)
        kMeansPredictions = (kMeanz.getPrediction())
        print("Starting Kmeans..")
        dataFrameGenerator(self.sampleDataset, kMeansPredictions, 5, self.splitFileName, "Kmeans")
        print(kMeansPredictions)
        pltPlotter(self.sampleDataset, kMeansPredictions, "Kmeans", 5, self.splitFileName)

    def startHeirarchical(self):
        print("Starting Heirarchical...")
        heirarchical = HeirarchialClustering(self.sampleDataset, 5)
        heirarchicalPredictions = heirarchical.getHeirarchical()
        dataFrameGenerator(self.sampleDataset, heirarchicalPredictions, 5, self.splitFileName, "Heirarchical")
        pltPlotter(self.sampleDataset, heirarchicalPredictions, "Heirarchical", 5, self.splitFileName)
        print("Finished Heirarchical...")

    def startSpectral(self):
        print("Starting Specrtral...")
        spectral = SpectralClustering(self.sampleDataset, 5)
        spectralPredictions = spectral.getSpectral()
        dataFrameGenerator(self.sampleDataset, spectralPredictions, 5, self.splitFileName, "Spectral")
        pltPlotter(self.sampleDataset, spectralPredictions, "Spectral", 5, self.splitFileName)
        print("Finished Spectral...")



manager = Controller("index_interactions.csv")



