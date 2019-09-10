'''
This modules responsible mainly to implementing solution answeing to 2.3Q and perform
Machine Learning algorithms and separate procedures.

Prepared by Vytautas Bielinskas
'''

# Import modules and packages
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 3.5


# Set system constants and global variables
N = 10    # Number of Possible Clusters


# Plot person activity in time
def plot_activity(data, user_number, this_class):
	'''
		Args:
			-- data : vector of number of travels in a given cluster for single person.
			-- user_number : number of user in a list.
			-- this_class : given index of cluster.

	'''

	data = data.reset_index()
	data['HH'] = data['HH'].apply(int)    # For better visualization

	# Find the Hour when the activity is the highest!
	hour_of_max = data.iat[data[data['Class']==np.max(data['Class'])].index.values.astype(int)[0], data.columns.get_loc('HH')]

	rcParams['figure.figsize'] = 12, 3

	_ = plt.title('Person {}, Cluster = {}'.format(user_number, this_class))
	_ = plt.fill_between(data['HH'], data['Class'], 0,
						 color='green', alpha=0.7,
						 edgecolor='black',
		                 label='Number of movements in hours')
	_ = plt.axhline(y=np.max(data['Class']), color='red', linestyle='--',
		            linewidth=4, label='Maximum activity')
	_ = plt.axvline(x=hour_of_max, color='red', linestyle='--',
		            linewidth=4, label='Hour of Maximum activity')
	_ = plt.grid(which='major', color='#cccccc', alpha=0.5)
	_ = plt.xticks(np.linspace(0, 23, 24))
	_ = plt.xlim(0, 23)
	_ = plt.ylim(0, np.max(data['Class'])+2)
	_ = plt.axvline(x=0, color='red')
	_ = plt.axvline(x=23, color='red')
	_ = plt.legend(shadow=True, loc='best')
	_ = plt.show()

	return hour_of_max


# Identify the location type by maximum activity hour
def identify_type_location(hour):
	'''
		Args:
			-- hour : hour of maximum activity for specific user in specific cluster (location)
	'''

	types = {0 : 'Home', 1 : 'Home', 2 : 'Home', 3 : 'Home', 4 : 'Home', 5 : 'Home', 6 : 'Home', 7 : 'Home', 8 : 'Home', 9 : 'Home', 10 : 'Home',
	         11 : 'Work', 12 : 'Work', 13 : 'Work', 14 : 'Work', 15 : 'Work', 16 : 'Work', 17 : 'Work', 18 : 'Work', 19 : 'Home', 20 : 'Home',
	         21 : 'Home', 22 : 'Home', 23 : 'Home'}

	return types.get(hour)


def identify_locations(user_number, data):
	'''
		Args:
			-- data : user data with clustered travels (classes)
			-- user_number : number of user in a list
	'''

	if len(data['Class'].unique()) > 0:
		for this_class in list(data['Class'].unique()):
			data_this_cluster = data[data['Class'] == this_class]

			min_timing = int(np.median(data_this_cluster['HH'].apply(float)))

			agg_time_report = data_this_cluster.groupby('HH')['Class'].count()
			# Plot the Person activity in time

			'''
			Note: I will find the hour of maximum activity by analyzing the plot of user activity
			      in a day where is hitorically aggregated data for specific user.
			'''

			hour_of_max_activity = plot_activity(agg_time_report, user_number, this_class)

			# Find the type of this location (cluster)
			type_location = identify_type_location(hour_of_max_activity)
			print('This location (Cluster {}) for Person {} is {}'.format(user_number, this_class, type_location.upper()))

			'''
			Note: for check the real location where the cluster taka place, please check the <this_user_data> DataFrame.	
			'''

	return None


def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2


# Plot Elbow method results
def plot_elbow_results(wscc, person, n_clusters):
	'''
		Args:
			-- wscc : vector of elbow values for a given data subset (training data)
	'''
	rcParams['figure.figsize'] = 14, 3.5

	_ = plt.plot(range(1, N), wscc, linewidth = 4, color = 'black',
	             marker = 'D', markersize = 7)
	_ = plt.axvline(x=n_clusters, color='red', linestyle='--', linewidth=4, label='Number of Locations')
	_ = plt.title('The Elbow Method for Person {}'.format(person), family = 'Arial', fontsize = 11, color = 'black')
	_ = plt.xlabel('The Number of Clusters',
	              family = 'Arial', fontsize = 9, color = 'black')
	_ = plt.ylabel('WCSS', family = 'Arial', fontsize = 9, color = 'black')
	_ = plt.xticks(fontsize = 9, color = 'black')
	_ = plt.yticks(fontsize = 9, color = 'black')
	_ = plt.grid(which = 'major', color = 'black', axis = 'x', alpha = 0.5)
	_ = plt.legend(shadow=True, loc='best')
	_ = plt.show()

	return None


# Plot scatter with clustered observations
def plot_clusters(X, n_clusters):
	'''
		Args:
			-- X : vector of data to be clustered
			-- n_clusters : number of optimum cluster in this dataset
	'''

	rcParams['figure.figsize'] = 5, 5

	kmeans = KMeans(n_clusters=n_clusters)
	kmeans.fit(X)
	y_kmeans = kmeans.predict(X)

	_ = plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, marker='+', cmap='viridis')

	centers = kmeans.cluster_centers_
	_ = plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='D', alpha=0.5)
	_ = plt.grid(which='major', alpha=0.5, color='#cccccc')
	_ = plt.show()

	return y_kmeans


# Remove meaningless classes
def remove_meaningless_classes(data):
	'''
		Args:
			-- data : data for the uses with predicted classes
	'''

	# Aggreate some data
	data_agg = data.groupby('Class')['latitude'].count()

	# Set contstants
	THRESHOLD = 25    # Travels

	classes_to_remove = []

	for this_class in data['Class'].unique():
		if len(data[data['Class'] == this_class]) < THRESHOLD:
			classes_to_remove.append(this_class)

	if len(classes_to_remove) > 0:
		for this_class in classes_to_remove:
			data = data[data['Class'] != this_class]
			print('(!) REMOVED meaningless/random/incorrect class = {}'.format(this_class))

	return data


# Find optimum number of clusters for a given data subset
def find_number_of_clusters(data):

	''' We are going to commpute the within cluster sum of squares for N
	 different numbers of clusters. '''

	# Generate data for clustering
	data = data[['latitude', 'longitude', 'Person', 'Departure', 'DAY_timestamp', 'HH', 'mm']]
	
	X = data.iloc[:, [0, 1]].values

	person = data.iat[0, data.columns.get_loc('Person')]

	# Normalize data
	#X = normalize(X)

	wscc = []

	for i in range(1, N):
	    
	    # 1. Fit the KMeans algorithm to our data X
	    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 10,
	                   n_init = 5, random_state = 42)
	    kmeans.fit(X)
	    
	    """
	    ::: KMeans args :::
	    max_iter : the maximum number of iteration there can be to find
	               the final clusters when KMeans algorithm is running.
	               
	    n_init   : the number of times the KMeans algorithm will be run
	               with different initial centroids.
	    """
	    
	    # 2. Compute the within cluster Sum of Squares and Append to
	    # our WSCC list (a.k.a. Inertia)
	    wscc.append(kmeans.inertia_)

	# 3. Plot Elbow results
	n_clusters = optimal_number_of_clusters(wscc)

	plot_elbow_results(wscc, person, n_clusters)
	predicted_classes = plot_clusters(X, n_clusters)

	# 4. Assign predicted classes to dataframe
	data['Class'] = predicted_classes
	#print(data.head(3))

	# 5. Remove meaningless classes
	data = remove_meaningless_classes(data)

	return data