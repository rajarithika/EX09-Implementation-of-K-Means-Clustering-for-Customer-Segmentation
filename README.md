# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data processing
2. Data preprocessing
3. initialize centroids
4. Asign clustures
5. Update centroid
 
 
  

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Raja rithika
RegisterNumber:  2305001029
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data


X=data[['Annual Income (k$)','Spending Score (1-100)']]

plt.figure(figsize=(4,4))
plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(X)
centroids=Kmeans.cluster_centers_


labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)

colors=['r','g','b']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/7e6601ad-b0f5-4438-9927-3cec2f858307)
![image](https://github.com/user-attachments/assets/bd6b43c0-3487-4474-88f7-a739745a0a2d)
![image](https://github.com/user-attachments/assets/aa47b86a-b447-40e0-b430-0935fdf0d808)
![image](https://github.com/user-attachments/assets/612e22e1-54bf-48de-bce1-0267407d53e7)







## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
