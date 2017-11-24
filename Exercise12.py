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

#set working directory
os.chdir('/Users/omneelay/Desktop/Exercise12/Intro_Biocomp_ND_318_Tutorial12/')

#read file
data=pd.read_csv('chickwts.txt')

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
