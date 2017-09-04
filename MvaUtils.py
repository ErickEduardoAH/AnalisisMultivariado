import numpy as np
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
    print('\t\tResume for '+name+'\nMaximum likehood estimators')
    print('-Mean vector (Mu):\n'+str(mu))
    print('-Covariance matrix (Sigma):\n'+str(sigma))
    print('\nSpectral decomposition')
    print('-EigenValues of Sigma:\n'+str(eigenValues))
    print('-EigenVectors matrix of Sigma (Gamma):\n'+str(eigenVectors))
    print('\nProbability quantile')
    print('-Chi-Squared critical value at 1-'+str(alpha)\
          +' percent: '+str(chiSquaredCriticalVale)+'\n\n')
    plotConfidenceEllipse(plt,mu,eigenValues,eigenVectors,chiSquaredCriticalVale,color=color,alpha=0.25)
    
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
    for i,l in enumerate(range(0,cols)):
        xs = [0,M[i,0]]
        ys = [0,M[i,1]]
        plt.plot(xs,ys)
    plt.plot(0,0,'ok')
    plt.axis('equal')
    plt.legend(colums)
    plt.grid(b=True, which='major')