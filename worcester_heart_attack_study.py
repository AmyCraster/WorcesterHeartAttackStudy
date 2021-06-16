import sksurv.datasets
from sksurv.datasets import load_whas500
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import ppscore as pps
import statistics
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import plot_lifetimes
from lifelines.statistics import logrank_test
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

data_x, data_y = load_whas500()
data_y_list = data_y.tolist()
data_y_dataframe=pd.DataFrame(data_y, columns = ["fstat", "lenfol"])
data = pd.concat([data_x, data_y_dataframe], axis=1)

#Check missing values
#print(data.isnull().sum())
#print(data.shape)

#Feature Engineering
#Check censored data
#fstat = True patients are regarded as censored data.  Make a new variable 'dead' that is 0 when fstat is false and 1 when fstat is True
data['dead'] = np.where(data['fstat']==True, 1,0)
data = data.drop(columns=['fstat'])
print(data.groupby('dead').count())
print(data)

#Check column types
print(data.info())

#Change categorical variables to numeric

categorical_columns = data.select_dtypes(include='category').columns
data[categorical_columns] = data[categorical_columns].astype('int64')
print(data.info())

#Look at correlation of variables through a heatmap
print(data.corr())
#Create visualisation of heatmap
plt.figure(figsize=(16,12))
sns.heatmap(data.corr(),annot=True,fmt=".2f")
plt.show()

#Drop features that are not significant - those with a correlation value close to zero
data = data.drop(columns = ['av3', 'cvd', 'los', 'sysbp'])

#Create an overall Kaplain Meier
durations = data['lenfol']
event_observed = data['dead'] # 1 if patient died, 0 if censored

km=KaplanMeierFitter()
km.fit(durations, event_observed, label='Kaplan Meier esimate')
km.plot()
plt.show()

#Age seems to have the strongest correlation with death, lets have a further look at this relationship
#Investigate difference in survival time based on age
print(statistics.median(data['age']))
#Create two age groups of those above 72 and those below 72
age_group_below_72 = data['age'] < statistics.median(data['age'])
age_group_above_72 = data['age'] > statistics.median(data['age'])
#Kaplain Meier for different ages
kmf = KaplanMeierFitter()
T = data['lenfol']
E = data['dead']
#Fit the data to the above 72 years group
kmf.fit(T[age_group_below_72], E[age_group_below_72], label = 'Patients younger than 72')
ax=kmf.plot()

kmf.fit(T[age_group_above_72], E[age_group_above_72], label='Patients older than 72')
ax1 = kmf.plot(ax=ax)
plt.show()

#Congestive heart complications  also had a very strong correlation with death
#Create the group
congestive_heart_complications = data['chf']==1
kmf2 = KaplanMeierFitter()
T = data['lenfol']
E = data['dead']
#Fit the data to those with congestive heart complications
kmf2.fit(T[congestive_heart_complications], E[congestive_heart_complications], label='Patients with chf')
ax=kmf2.plot()
#Fit the data to those without congestive heart complications
kmf2.fit(T[~congestive_heart_complications], E[~congestive_heart_complications], label= 'Patients without chf')
ax1 = kmf2.plot(ax=ax)
plt.show()

#Apply statistical tests to the data
#Use a logrank test to determine whether there is a statistical significance for age
results_age = logrank_test(T[age_group_below_72], T[age_group_above_72], event_observed_A = E[age_group_below_72], event_observed_B=E[age_group_above_72])
results_age.print_summary()

#Use a logrank test to determine whether there is a statistical significance for chf
results_chf = logrank_test(T[congestive_heart_complications], T[~congestive_heart_complications], event_observed_A = E[congestive_heart_complications], event_observed_B=E[~congestive_heart_complications])
results_chf.print_summary()


#Cluster the data - use unsupervised learning to cluster patients
#I decided to standardise the data and put all the features on the same scale because clustering uses distance metrics
X = data.drop('dead', axis=1)
X = data.values
X_std = preprocessing.StandardScaler().fit_transform(X)


#Choose number of clusters using wcss
# finding wcss value for different number of clusters

wcss = []

for i in range(1,6):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X_std)

  wcss.append(kmeans.inertia_)

# plot an elbow graph
f, ax = plt.subplots(figsize=(1, 6))
sns.set()
plt.plot(range(1,6), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#Suggests that using three clusters is optimum
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X_std)

print(Y)

#Add cluster label as an attribute to gain insights from it
data['cluster_num'] = Y 
print(data.head())

#group by the cluster value to see the mean value of each of the attributes in the dataset
print(data.groupby('cluster_num').mean())

#Look at the distribution of customers based on their age in their cluster
f, ax = plt.subplots(figsize=(1, 5))
sns.boxplot(y='age', x='cluster_num', 
                 data=data, 
                 palette="colorblind")
plt.xlabel('Cluster Number')
plt.title('Age variation across clusters')
plt.show()

#Look at the distribution of customers mortality based on their cluster
f, ax = plt.subplots(figsize=(1, 5))
sns.boxplot(y='dead', x='cluster_num', 
                 data=data, 
                 palette="colorblind")
plt.xlabel('Cluster Number')
plt.title('Death variation across clusters')
plt.show()

#Use survival analysis to find out which clinical features are important
# Use Cox proportional hazard model
cph = CoxPHFitter()
cph.fit(data, "lenfol", event_col="dead")
cph.print_summary()

#Create plot
cph.plot()
plt.show()

#Use another method to study which features are important
#Use logistic regression for this

y = data['dead'].values
model = LogisticRegression()
# fit the model
model.fit(X_std, y)
# get importance
importance = model.coef_[0]
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
print(np.std(X_std, 0)*model.coef_[0])

plt.bar(["afb", "age", "bmi", "chf", "diaspb", "gender", "hr", "miord", "mitype", "sho", "lenfol","cluster number"], importance)
#plt.bar(["afb", "age"], importance[:2])
plt.title("Feature importa")
plt.xlabel('Magnitude of coefficient')
plt.show()
