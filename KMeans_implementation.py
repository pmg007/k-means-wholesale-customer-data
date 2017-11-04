import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')
X = np.array([[1,2],
			  [1.5,1.8],
			  [5,8],
			  [8,8],
			  [1,0.6],
			  [9,11]
			])

#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()

colors = 10*["g","r","c","b","k"]

class K_Means:
	"""docstring for K_Means"""
	def __init__(self, k=2, tol=0.001, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		# empty dictionary of centroids
		self.centroids = {}

		# assigning first k values in dataset to be centroids, though centroids may be selected randomly
		for i in range(self.k):
			self.centroids[i] = data[i]

		# loop which runs max_iter times to check if centroid are in tolerance range or not
		for i in range(self.max_iter):
			self.classifications = {}

			for i in range(self.k):
				self.classifications[i] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids = dict(self.centroids)

			for classification  in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)

			#check if we have reached optimized stage
			optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
					print (np.sum((current_centroid-original_centroid)/original_centroid*100.0))
					optimized = False

			if optimized:
				break


	def predict(self, data):
		distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

# end of class

# using the class for clustering

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color = "k", s=150, linewidth=5)

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker = "x", color=color, s=150, linewidth=5)

plt.show()