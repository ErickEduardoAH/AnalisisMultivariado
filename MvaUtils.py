import numpy as np
import pandas as pd
from scipy.stats import chi2
from matplotlib.patches import Ellipse
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.stat import Statistics
from pyspark.ml.linalg import Vectors
from pyspark.sql import Column as c
from pyspark.sql.functions import array, udf, lit, col as c
from pyspark.sql import types

def computeMaximumLikelihoodEstimators(dataSetDF):
    mu = np.array(dataSetDF.groupby().mean().rdd.map(lambda r: r[:]).top(1)[0])
    sigma = RowMatrix(dataSetDF.rdd.map(lambda r: [r[:]])).computeCovariance().toArray()
    return mu, sigma

def diagonalize(sigma):
    eigenValues, eigenVectors = np.linalg.eigh(sigma)
    order = eigenValues.argsort()
    return eigenValues, eigenVectors

def plotConfidenceEllipse(plt,mu,eigenValues,eigenVectors,chiSquaredCriticalVale,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    theta = np.degrees(np.arctan2(*eigenVectors[:,0][::-1]))
    width,height = chiSquaredCriticalVale*np.sqrt(eigenValues)
    confidenceEllipse = Ellipse(xy=mu,width=width,height=height,angle=theta,**kwargs)
    ax.add_artist(confidenceEllipse)
    return confidenceEllipse

def getProbabilityDensityContour(plt,datasetDF,cols,alpha,freedomDegrees,color,name='DataSet'):
    mu, sigma = computeMaximumLikelihoodEstimators(datasetDF.select(cols))
    eigenValues, eigenVectors = diagonalize(sigma)
    chiSquaredCriticalVale = chi2.ppf(q=(1-alpha),df=freedomDegrees)
    plotConfidenceEllipse(plt,mu,eigenValues,eigenVectors,chiSquaredCriticalVale,color=color,alpha=0.25)
    sumaryTable=[]
    sumaryTable.append(['Mean',mu])
    sumaryTable.append(['Covariance matrix',sigma])
    sumaryTable.append(['EigenValues',eigenValues])
    sumaryTable.append(['EigenVectors',eigenVectors])
    sumaryTable.append(['Confidence',1-alpha])
    sumaryTable.append(['Chi-squared critical value',chiSquaredCriticalVale])
    sumaryTableDF = pd.DataFrame(sumaryTable,columns=['Summary',name])
    return sumaryTableDF
    
def scatterPlot(plt,dataDF,col_1,col_2,color):
    dataPD = dataDF.select([col_1,col_2]).toPandas()
    plt.scatter(dataPD[col_1],dataPD[col_2],color=color)
    
def getCanonicalBase(dim):
    canonicalBase = []
    for i in range(0,dim):
        e = [float(0)]*4
        e[i]=1
        canonicalBase.append(Vectors.dense(e))
    return canonicalBase

def plotProjectedBase(plt,components,projectedCanonicalsPD,colums):
    M = projectedCanonicalsPD[components].as_matrix()
    rows,cols = M.T.shape
    maxes = np.amax(abs(M), axis = 0)
    t = np.linspace(-100, 100, 100)
    for i,l in enumerate(range(0,cols)):
        plt.plot(t*M[i,0],t*M[i,1],color='blue',alpha=0.3)
        plt.axes().arrow(0,0,M[i,0],M[i,1],head_width=0.04,head_length=0.1,color='black')
        plt.annotate(colums[i],xy=(M[i,0],M[i,1]),xytext=(M[i,0]+3/100,M[i,1]+3/100))
    plt.plot(0,0,'ok')
    plt.grid(b=True, which='major')