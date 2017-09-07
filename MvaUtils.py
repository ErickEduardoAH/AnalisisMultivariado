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

def transformCanonicalBase(plt,components,canonicalBaseDF,colums):
    M = 3*canonicalBaseDF.select(components).toPandas().as_matrix()
    rows,cols = M.T.shape
    maxes = 1.1*np.amax(abs(M), axis = 0)
    for i,l in enumerate(range(0,cols)):
        plt.axes().arrow(0,0,M[i,0],M[i,1],head_width=0.05,head_length=0.1,color = 'black')
    plt.plot(0,0,'ok') #<-- plot a black point at the origin
    plt.axis('equal')  #<-- set the axes to the same scale
    plt.xlim([-maxes[0],maxes[0]]) #<-- set the x axis limits
    plt.ylim([-maxes[1],maxes[1]]) #<-- set the y axis limits
    plt.grid(b=True, which='major') #<-- plot grid lines