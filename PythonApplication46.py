#Dimensional Reduction
#PCA
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import preprocessing

iris=pd.read_csv("iris.csv")
print(iris)

p=preprocessing.LabelEncoder()
iris.variety=p.fit_transform(iris.variety)

X=iris.iloc[ : , 0:4 ].values
Y=iris.iloc[ : , 4].values

pca=decomposition.PCA(n_components=3)
pca.fit(X)

print(X[: 10])
print("-----------------------")

X=pca.transform(X)
print(X[: 10])

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")

ax.scatter(X[Y==0, 0],X[Y==0, 1],X[Y==0, 2],c="red")
ax.scatter(X[Y==1 , 0],X[Y==1 , 1],X[Y==1 , 2],c="blue")
ax.scatter(X[Y==2 , 0],X[Y==2 , 1],X[Y==2 , 2],c="green")

plt.show()