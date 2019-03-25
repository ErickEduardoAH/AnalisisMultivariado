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
from scipy.stats import norm
from numpy import linalg as linear
from numpy import random as ran
from scipy.stats import norm

def computeMaximumLikelihoodEstimators(dataSetDF):
    mu = np.array(dataSetDF.groupby().mean().rdd.map(lambda r: r[:]).top(1)[0])
    sigma = RowMatrix(dataSetDF.rdd.map(lambda r: [r[:]])).computeCovariance().toArray()
    return mu, sigma

def diagonalize(sigma):
    eigenValues, eigenVectors = np.linalg.eigh(sigma)
    index = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[index]
    eigenVectors = eigenVectors[:,index]
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
    
def scatterPlot(plt,dataDF,col_1,col_2,color,labels=None):
    dataPD = None
    if labels == None:
        dataPD = dataDF.select([col_1,col_2]).toPandas()
    else:
        dataPD = dataDF.select([col_1,col_2,labels]).toPandas()
        names = dataPD[[labels]].as_matrix().transpose()[0]            
        for i, name in enumerate(names):
            plt.annotate(name, (dataPD[col_1][i]+0.1,dataPD[col_2][i]+0.1),color=color)
    plt.scatter(dataPD[col_1],dataPD[col_2],color=color)
    return plt
    
def getCanonicalBase(dim):
    canonicalBase = []
    for i in range(0,dim):
        e = [float(0)]*4
        e[i]=1
        canonicalBase.append(Vectors.dense(e))
    return canonicalBase

def plotProjectedBase(plt,components,projectedCanonicalsPD,colums,axesColor='blue',axesAlpha=1,arrowWidth=1.0,\
                      arrowColor='black',arrowAlpha=1,head_width=0.04,head_length=0.1,shift=2/100):
    
    M = projectedCanonicalsPD[components].as_matrix()
    rows,cols = M.T.shape
    maxes = np.amax(abs(M), axis = 0)
    t = np.linspace(-1000, 1000, 100)
    plt.xlabel(components[0])
    plt.ylabel(components[1])
    plt.axvline()
    plt.axhline()
    for i,l in enumerate(range(0,cols)):
        plt.axes().arrow(0,0,M[i,0],M[i,1],head_width=head_width,head_length=head_length,\
                         color=arrowColor,alpha=arrowAlpha,linewidth=arrowWidth)
        plt.annotate(colums[i],xy=(M[i,0],M[i,1]),xytext=(M[i,0]+shift,M[i,1]+shift),\
                         weight='bold',color=arrowColor)
        plt.plot(t*M[i,0],t*M[i,1],color=axesColor,alpha=axesAlpha)        
    plt.plot(0,0,'ok')
    plt.grid(b=True, which='major')

def mvNormalizerMatrix(Sigma):
    eigenValues, eigenVectors = diagonalize(Sigma)
    LambdaSqrt = np.diag([np.sqrt(v) for v in eigenValues])
    return np.dot(LambdaSqrt,eigenVectors), eigenValues, eigenVectors

def multivariateNormalVector(X,Shift,LinearTransform):
    Z = [one*norm.ppf(ran.uniform()) for one in np.ones(len(Shift))]
    X = [float(x) for x in np.dot(Z,LinearTransform)]
    for i in range(0,len(X)): X[i]=X[i]+Shift[i]
    return X