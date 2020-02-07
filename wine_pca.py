### PCA Wine ###
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from scipy.stats import skew, kurtosis
import seaborn as sns

wine = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\PCA\wine.csv")
wine.describe()
wine.head()
wine.shape
wine.drop_duplicates(keep='first',inplace=True)
wine.shape
wine.columns
wine.isna().sum()
wine.isnull().sum()

np.mean(wine.Type) 
np.median(wine.Type) 
# despersion
np.std(wine)
np.var(wine.Type)
np.std(wine.Type) 

plt.hist(wine['Malic']);plt.xlabel('Malic');plt.ylabel('Type');plt.title('histogram of Malic')
plt.hist(wine['Alcohol']);plt.xlabel('Alcohol');plt.ylabel('Type');plt.title('histogram of Alcohol')
plt.hist(wine['Ash']);plt.xlabel('Ash');plt.ylabel('Type');plt.title('histogram of Ash')
plt.hist(wine['Alcalinity']);plt.xlabel('Alcalinity');plt.ylabel('Type');plt.title('histogram of Alcalinity')
plt.hist(wine['Magnesium']);plt.xlabel('Magnesium');plt.ylabel('Type');plt.title('histogram of Magnesium')
plt.hist(wine['Phenols']);plt.xlabel('Phenols');plt.ylabel('Type');plt.title('histogram of Phenols')
plt.hist(wine['Flavanoids']);plt.xlabel('Flavanoids');plt.ylabel('Type');plt.title('histogram of Flavanoids')
plt.hist(wine['Nonflavanoids']);plt.xlabel('Nonflavanoids');plt.ylabel('Type');plt.title('histogram of Nonflavanoids')
plt.hist(wine['Proanthocyanins']);plt.xlabel('Proanthocyanins');plt.ylabel('Type');plt.title('histogram of Proanthocyanins')
plt.hist(wine['Color']);plt.xlabel('Color');plt.ylabel('Type');plt.title('histogram of Color')
plt.hist(wine['Hue']);plt.xlabel('Hue');plt.ylabel('Type');plt.title('histogram of Hue')
plt.hist(wine['Dilution']);plt.xlabel('Dilution');plt.ylabel('Type');plt.title('histogram of Dilution')
plt.hist(wine['Proline']);plt.xlabel('Proline');plt.ylabel('Type');plt.title('histogram of Proline')

skew(wine)
kurtosis(wine)
np.mean(wine)

sns.pairplot((wine),hue='Type')
sns.heatmap(wine.corr(), annot=True) # with annot=True values are seen of specific realtion

from sklearn.decomposition import PCA

# Considering only numerical data 
wine.data = wine.ix[:,1:]
wine.data.head(15)
# Normalizing the numerical data 
wine_normal = scale (wine.data)
pca = PCA (n_components = 13)
pca_values = pca.fit_transform(wine_normal)
# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]
# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
# Variance plot for PCA components obtained 
plt.plot(var1,color= "red")
# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:2:3]
plt.scatter(x,y,color=["red","blue"])
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x, y, z, c='r', marker='o')
#ax.set_xlabel('x Label')
#ax.set_ylabel('y Label')
#ax.set_zlabel('z Label')

#plt.show()
#or
##Axes3D.scatter(np.array(x),np.array(y),np.array(z),c=["green","blue","red"])

################### Clustering  ##########################

from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 

X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
df_xy =pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y
df_xy.plot(x="X",y = "Y",kind="scatter")
model1 = KMeans(n_clusters=3).fit(df_xy)
model1.labels_
model1.cluster_centers_
df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)

def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

df_norm = norm_func(wine.iloc[:,1:])

df_norm.head(10)
new_df = pd.DataFrame(pca_values[:,0:4])

kmeans = KMeans(n_clusters = 5)
kmeans.fit(new_df)
kmeans.labels_
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine['clust']=md # creating a  new column and assigning it to new column 
wine.head()


wine1 = wine.iloc[:,[8,0,1,2,3,4,5,6,7]]

wine1.iloc[:,1:8].groupby(wine.clust).median()

wine1.to_csv("wine.csv")

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(12, 7));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('wine');plt.ylabel('Type')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(h_complete.labels_)

wine['clust']=cluster_labels # creating a  new column and assigning it to new column 
wine = wine.iloc[:,[8,0,1,2,3,4,5,6,7]]
wine.head()
# getting aggregate mean of each cluster
wine.iloc[:,2:].groupby(wine.clust).median()
# creating a csv file 
wine.to_csv("wine.csv",encoding="utf-8")

