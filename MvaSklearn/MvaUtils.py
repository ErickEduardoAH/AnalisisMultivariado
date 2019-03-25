import numpy as np
import pandas as pd
from scipy.stats import chi2
from matplotlib.patches import Ellipse

def computeMaximumLikelihoodEstimators(datasetPD,columns):
    mu =  list(datasetPD[columns].mean())
    sigma = np.matrix(datasetPD[columns].cov())
    return mu, sigma

def diagonalize(sigma):
    eigenValues, eigenVectors = np.linalg.eigh(sigma)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues, eigenVectors

def plotConfidenceEllipse(plt,mu,eigenValues,eigenVectors,chiSquaredCriticalVale,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    theta = np.degrees(np.arctan2(*eigenVectors[:,0][::-1]))
    width,height = chiSquaredCriticalVale*np.sqrt(eigenValues)
    confidenceEllipse = Ellipse(xy=mu,width=width,height=height,angle=theta,**kwargs)
    ax.add_artist(confidenceEllipse)
    return confidenceEllipse

def getProbabilityDensityContour(plt,datasetPD,columns,alpha,freedomDegrees,color,name='DataSet'):
    mu, sigma = computeMaximumLikelihoodEstimators(datasetPD,columns)
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

def scatterPlot(plt,datasetPD,col_1,col_2,colorDots='darkblue',label=None,shift=0.05,**kwargs):
    dataPD = datasetPD[[col_1,col_2]]
    plt.scatter(dataPD[col_1],dataPD[col_2],color=colorDots)
    if(label!=None):
        points = datasetPD[[col_1,col_2,label]].values
        for point in points:
            plt.text(point[0]+shift,point[1]+shift,str(point[2]))
    plt.axvline(**kwargs)
    plt.axhline(**kwargs)
    
def getMvNormalSample(mu,sigma,n):
    columns = ['x'+str(i+1) for i in range(0,len(mu))]
    return pd.DataFrame(np.random.multivariate_normal(mu,sigma,n),columns=columns)
    
def getCenteredDataMatrix(dataPD,columns,keys):
    dataframePD = dataPD.copy(deep=True)
    meanVector = dataPD.mean().to_dict()
    for col in columns:
        dataframePD['centered_'+col] = dataframePD[col]-meanVector[col]
    return dataframePD[keys+['centered_'+col for col in columns]]

def overlapedHitograms(plt,datasetPD,columns,target,labels=None,sample=1.0,bins=10,size=(17,8),alpha=0.7):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=size)
    if labels == None:
        labels = [0,1]
    negativesPD = datasetPD[datasetPD[target]==labels[0]]
    positivesPD = datasetPD[datasetPD[target]==labels[1]]
    for i, column in enumerate(columns):
        negativeValues = negativesPD[column].values.transpose()
        positiveValues = positivesPD[column].values.transpose()
        ax = fig.add_subplot(2, 2, i+1,)
        ax.set_title('discriminant power '+column)
        ax.hist(negativeValues,bins=bins,alpha=alpha)
        ax.hist(positiveValues,bins=bins,alpha=alpha)

def plotProjectedBase(plt,components,projectedCanonicalsPD,colums,axesColor='blue',axesAlpha=1,arrowWidth=1.0,\
                      arrowColor='black',arrowAlpha=1,head_width=0.04,head_length=0.1,shift=2/100):
    
    M = projectedCanonicalsPD[components].as_matrix()
    rows,cols = M.T.shape
    maxes = np.amax(abs(M), axis = 0)
    t = np.linspace(-1000, 1000, 100)
    plt.xlabel(components[0])
    plt.ylabel(components[1])
    for i,l in enumerate(range(0,cols)):
        plt.axes().arrow(0,0,M[i,0],M[i,1],head_width=head_width,head_length=head_length,\
                         color=arrowColor,alpha=arrowAlpha,linewidth=arrowWidth)
        plt.annotate(colums[i],xy=(M[i,0],M[i,1]),xytext=(M[i,0]+shift,M[i,1]+shift),\
                         weight='bold',color=arrowColor)
        plt.plot(t*M[i,0],t*M[i,1],color=axesColor,alpha=axesAlpha)        
    plt.plot(0,0,'ok')
    plt.grid(b=True, which='major')
    
def screePlot(plt,ratios):
    display(pd.DataFrame(ratios,columns=['explained variance']))
    plt.bar(list(range(1,len(ratios)+1)),ratios,color='Blue')