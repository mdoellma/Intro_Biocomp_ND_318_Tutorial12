#Exercise 12 Question 1
###############################################################################################################
##Generating a plot

#Load packages
import numpy
import pandas
from plotnine import *

#Load file
chicks=pandas.read_csv("chickwts.txt", header=0, sep=",")

#Generate bar graph that summarizes means of data
d=ggplot(chicks)+theme_classic()+xlab("feed")+ylab("weight")
d+geom_bar(aes(x="factor(feed)",y="weight"),stat="summary",fun_y=numpy.mean)

##############################################################################################################
##Running the likelihood ratio test 

#Import packages
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

#Subsetting data so it only includes sunflower & soybean data points
SFvsSB=chicks.loc[chicks.feed.isin(['sunflower', 'soybean']),:]

#Creating a new dataframe to change x column to 0's and 1's
SFvsSBdf=pandas.DataFrame({'y':SFvsSB.weight, 'x':0})
SFvsSBdf.loc[SFvsSB.feed=='sunflower', 'x']=1

#Null hypothesis likelihood ratio equation
def nllike(p,obs):
    B0=p[0]
    sigma=p[1]
    expected=B0
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
    
#Alternative hypothesis likelihood ratio equation
def nllike2(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

#Estimating parameters by minimizing the nll 
initialVals1=numpy.array([1,1,1])

#Fit null and alternative hypotheseses to Nelder-Mead model
fitNull=minimize(nllike,initialVals1, method="Nelder-Mead",options={'disp': True}, args=SFvsSBdf)
fitAlt=minimize(nllike2,initialVals1, method="Nelder-Mead",options={'disp': True}, args=SFvsSBdf)
print(fitNull.x)
print(fitAlt.x)

#Calculating p-value
from scipy.stats import chi2
D=(2*(fitNull.fun-fitAlt.fun))
feedanswer=(1-chi2.cdf(x=D,df=1))
print('weight p value')
print(feedanswer)





