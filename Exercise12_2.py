##Exercise12

##var1=re.compile(regexstring)
##list1=[list of strings to search]
##filter(var1.match, list1)

#import packages
import os
import numpy as np
import pandas as pd
import scipy
import sklearn
import scipy.integrate as spint
from plotnine import *
from ggplot import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

#set working directory
#os.chdir('/Users/omneelay/Desktop/Exercise12/Intro_Biocomp_ND_318_Tutorial12/')
os.chdir('C:\\Users\\jsh\\OneDrive\\github\\BioComp\\Intro_Biocomp_ND_318_Tutorial12\\')

#read file
data=pd.read_csv('chickwts.txt')

#PART 1
horsebean_avg=data[['weight']].where(data[['feed']].values=='horsebean').stack().mean()
linseed_avg=data[['weight']].where(data[['feed']].values=='linseed').stack().mean()
soybean_avg=data[['weight']].where(data[['feed']].values=='soybean').stack().mean()
sunflower_avg=data[['weight']].where(data[['feed']].values=='sunflower').stack().mean()
meatmeal_avg=data[['weight']].where(data[['feed']].values=='meatmeal').stack().mean()
casein_avg=data[['weight']].where(data[['feed']].values=='casein').stack().mean()

avg_list=[horsebean_avg,linseed_avg,soybean_avg,sunflower_avg,meatmeal_avg,casein_avg]
feed_list=[1,2,3,4,5,6]

LABELS = ['horsebean', 'linseed', 'soybean', 'sunflower', 'meatmeal', 'casein']

plt.bar(feed_list, avg_list, align='center')
plt.xticks(feed_list, LABELS)
plt.show()


##PART 2
searchfor = ['horsebean', 'linseed', 'meatmeal', 'casein', 'soybean']
sunflowerdata = data[~data.feed.str.contains('|'.join(searchfor))]
searchfor = ['horsebean', 'linseed', 'meatmeal', 'casein', 'sunflower']
soybeandata=data[~data.feed.str.contains('|'.join(searchfor))]

#making data frame x and y where x=0 for sunflower and x=1 for soybean
sunflowerdata=sunflowerdata.weight
sunflowerdata=sunflowerdata.values.tolist()
a=[0,0,0,0,0,0,0,0,0,0,0,0]
sunflowerdata=pd.DataFrame({'x': a, 'y': sunflowerdata})

soybeandata=soybeandata.weight
soybeandata=soybeandata.values.tolist()
a=[1,1,1,1,1,1,1,1,1,1,1,1,1,1]
soybeandata=pd.DataFrame({'x': a, 'y': soybeandata})
part2data=soybeandata.append(sunflowerdata)

#likelihood fuctions

def nllikealt(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
    
def nllikenull(p,obs):
    B0=p[0]
    sigma=p[1]
    
    expected=B0
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
    
initialGuessnull=np.array([1,1])
initialGuessalt=np.array([1,1,1])

#minimizing the nll functions null and alt. And calculating D.
fit=minimize(nllikenull,initialGuessnull,method="Nelder-Mead",options={'disp': True},args=part2data)
part2_null=fit.fun
fit=minimize(nllikealt,initialGuessalt,method="Nelder-Mead",options={'disp': True},args=part2data)
part2_alt=fit.fun
part2_D=2*(part2_null-part2_alt)

#getting p value and testing for significance
print("p-value part2: ",1-scipy.stats.chi2.cdf(x=part2_D,df=1)," SIGNIFICANT")