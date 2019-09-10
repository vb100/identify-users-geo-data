
'''
Main file dedicated to Data ANalysis functions and procedures for the Sentiance Task.

MUST BE IMPORTED TO MAIN JUPYTER NOTEBOOK FILE.

Prepared by Vytautas Bielinskas. vytautas.bielinskas@gmail.com
https://www.linkedin.com/in/bielinskas/, https://www.youtube.com/c/VytautasBielinskas 
'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Import modules and packages
import sys
import numpy as np
import pandas as pd
import os
from os import listdir
from custom_modules_vb import colors as C
from custom_modules_vb import ecdf as E
from datetime import datetime, date, time, timedelta
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 5, 5


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# System constants
SEPARATOR = '{}>'.format('-' * 30)

# Check if a Python file is running inside any virtual environment
def is_venv():
	'''
		Args: None
	'''
	return (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


# Generate a map with a given lists of coordinates (latitudes on x and longitudes on y axis)
def map_person(latitudes, longitudes, person_id):
	'''
		Args:
			latitudes : a given list of latitudes of a person has visited.
			longitudes : a given list of latitudes of a person has visited.
	'''

	_ = plt.title('Activity of Person {}'.format(person_id), family='Arial', fontsize=11)
	_ = plt.plot(latitudes, longitudes,
				 label='Visited points',
				 linestyle=None, linewidth=0,
				 marker='.',
				 markersize='5',
				 alpha=0.9,
				 color='#128128')
	_ = plt.grid(which='major', color='#cccccc', alpha=0.5)
	_ = plt.xlabel('Latitudes', family='Arial', fontsize=9)
	_ = plt.ylabel('Longitues', family='Arial', fontsize=9)
	_ = plt.show()

	return None


# Read all datasets in project directory
def read_data(filelist, DIR):
	'''
		Args:
			-- filelist : a list (type) of files that contain a data for this project in given file format.
			-- DIR - current working directory (type: string)
	'''

	# Change working directory to a <data> folder
	os.chdir('./{}'.format(DIR))
	print('Directory changed to: {}\n'.format(os.getcwd()))

	# List of files
	files = []    # Empty list at the very begining

	for i, this_file in enumerate(filelist):
		print('| Reading: {}'.format(this_file))
		temp_df = pd.read_csv(this_file, sep=';')
		# Add index of person as an extra column in a dataset
		temp_df['Person'] = i
		files.append(temp_df)
		print('|| File with shape of {} is appended to the main list.'.format(temp_df.shape))
		print('{}\n'.format(temp_df.head(3)))
		
		# Plot the small map
		map_person(temp_df['latitude'], temp_df['longitude'], i)

	# Merge all found datasets into one single Pandas DataFrame
	data = pd.concat(files)
	print('All datasets connected into single one DataFrame with shape of {}.'.format(data.shape))

	return data


# Find all datasets with a given data format
def find_datasets(FORMAT):
	'''
		Args: 
			-- FORMAT : a data format that a function will find all dataset (type: string)
	'''
	print('{} SEARCHING AND READING DATA'.format(SEPARATOR))


	cwd = os.getcwd()
	print('Searching for datasets in project directory: {}'.format(cwd))

	# Set Constants
	DIR = 'data'

	# The list of found dataset within project directory
	filenames = listdir('./{}'.format(DIR))
	n = len(filenames)

	# Some output
	if n > 0:
			print('Found files: {}'.format(list(filenames)))
			print('Number of found files: {}.'.format(n))
			data = read_data(filenames, DIR)

	else:
		print('Found zeros valid files.')
		data = None

	return data


# Parse Timestamp column
# ::: Custom function to extract separate elements of a given timestamp
def get_years(x):
	return x[:4]

def get_month(x):
	return x[4:][:2]

def get_day(x):
	return x[6:][:2]

def get_hours(x):
	return x[8:][:2]

def get_minutes(x):
	return x[10:][:2]

def get_offset(x):
	if '-' in x:
		return int(int((-1) * int(x.split('-')[1]))/100)
	elif '+' in x:
		return int(int(x.split('+')[1])/100)


# Convert separate elements to single timestamp
def convert_to_timestamp(x):
	'''
		Args:
			x : list of separate timestamp elements in <string>
	'''
	if '-' in str(x['OFFSET']):
		offset_ = int(str(x['OFFSET']).replace('-', ''))
		return datetime.strptime('{}-{}-{}-{}-{}'.format(x['YEAR'], x['MONTH'], x['DAY'], x['HH'], x['mm']), '%Y-%m-%d-%H-%M') - timedelta(hours=offset_)
	else:
		offset_ = int(x['OFFSET'])
		return datetime.strptime('{}-{}-{}-{}-{}'.format(x['YEAR'], x['MONTH'], x['DAY'], x['HH'], x['mm']), '%Y-%m-%d-%H-%M') + timedelta(hours=offset_)


# Parse dirty timestamo string to readable format
def parse_timestamp(data, c_timestamp, c_duration):
	'''
		Args:
			-- data : a given dataframe
			-- c_timestamp : a name <string> of a column that need to be parsed
			-- c_duration : a name <string> of a column that need to be included into parsing

		Example:
			start_time(YYYYMMddHHmmZ) = 201312251147-0300 --> 2013(YYYY) 12(MM) 25(dd) 11(HH) 47(mm) 0300(Z)
	'''

	# Extract separate values from the column on whole dataset


	print('{} PARSING DATA'.format(SEPARATOR))

	if c_timestamp in list(data.columns) and c_duration in list(data.columns):
		data['YEAR'] = data[c_timestamp].apply(get_years)
		data['MONTH'] = data[c_timestamp].apply(get_month)
		data['DAY'] = data[c_timestamp].apply(get_day)
		data['HH'] = data[c_timestamp].apply(get_hours)
		data['mm'] = data[c_timestamp].apply(get_minutes)
		data['OFFSET'] = data[c_timestamp].apply(get_offset)

		data['Departure'] = data.apply(convert_to_timestamp, axis=1)

		data = data.drop([c_timestamp], axis=1)
	
	else:
		print('Parsing is not allowed')

	return data

# Aggregate by day (function)
def aggregate_day_column(x):
	'''
		Args:
			-- x : list of separate timestamp elements in <string>
	'''
	return datetime.strptime('{}-{}-{}'.format(x['YEAR'], x['MONTH'], x['DAY']), '%Y-%m-%d')


# Aggregate by day (caller)
def aggregate_by_day_per_user(data):
	'''
	Args:
		-- data : full Pandas DataFrame with parsed data
	'''

	# Create a fake counter for an observation
	data['count'] = 1

	# Get a timestamp at <DAY> scale
	data['DAY_timestamp'] = data.apply(aggregate_day_column, axis=1)
	data = data[['Person', 'DAY_timestamp', 'count']]

	# Aggregate by user and calculate number of trips for each person per day
	data = data.groupby(['Person', 'DAY_timestamp'])['count'].count().unstack().T.reset_index()

	return data


# Calculate Average Number of Places visited per day
def calculate_avg_places(data):
	'''
		Args:
			-- data : full Pandas DataFrame with parsed data 
	'''

	for this_person in list(data.columns):
		if type(this_person) != str:
			avg_places = np.round(np.nanmean(data[int(this_person)].values), 3)
			print('| Person {} = {} visited places per day.'.format(this_person, (avg_places)))

	return None


# Calculate Euclide distance
def calculate_distance(x1, y1, x2, y2):
	'''
		Args:
			-- x1 : geographical longitude of the first object.
			-- y1 : geographical latitude of the second object.
			-- x2 : geographical longitude of the first object.
			-- y2 : geographical latitude of the second object.
	'''

	return np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))


# Calculate median distances traveled between two subsequent stationary locations for each person
def calculate_median_sub_distances(data, DISTANCE_RATIO):
	'''
		Args:
			-- data : parsed dataframe <Pandas DataFrame> 
	'''

	# Calculate number of users in a given dataframe
	n = len(data['Person'].unique())

	# Set helping indexes
	index_lat = data.columns.get_loc('latitude')
	index_lng = data.columns.get_loc('longitude')

	if n > 0:

		print('Median distances between two subsequent stationary locations for all active period are:')
		l = []    # List for storage all aggregated data for all users
		l_median_distances = []    # List of calculated median values of distances for whole period for each user

		for person in range(0, n):

			df_person = data[data['Person'] == person]

			# List of all distances for this user
			median_distances_USER_FULL = []

			# Add extra column - reserve it for distances in meters
			df_person['Distance'] = None
			index_distance = df_person.columns.get_loc('Distance')

			# Create a new empty list for this person to storage all aggregated distances by a day
			l_person = []

			all_timestamps = list(df_person['DAY_timestamp'].unique())       # The order is timestamps is not important!

			for this_timestamp in all_timestamps:

				df_person_this_timestamp = df_person[df_person['DAY_timestamp'] == this_timestamp]

				person_travels = {}    # I created this dictionary to storage day timestamps and median distances for each timestamp.

				if len(df_person_this_timestamp) > 1:    # If there are more than one travel per a day

					# I wanna be sure that timestamps are ordered in correct sequence (from the earlier to the lastest)
					df_person_this_timestamp = df_person_this_timestamp.sort_values(by=['Departure'], ascending=True)

					for this_travel in range(0, len(df_person_this_timestamp)-1):

						# Calculate the real distance between two stops in kilometers
						distance = calculate_distance(df_person_this_timestamp.iat[this_travel, index_lng], df_person_this_timestamp.iat[this_travel, index_lat],\
							df_person_this_timestamp.iat[this_travel+1, index_lng], df_person_this_timestamp.iat[this_travel+1, index_lat])
						df_person_this_timestamp.iat[this_travel, index_distance] = distance * DISTANCE_RATIO    # meters

						# Add calculated distance to non-aggregate list for this user
						if distance * DISTANCE_RATIO > 0.05:    # Including those distances that are bigger than 50 meters. Other ones are meaningless.   
							median_distances_USER_FULL.append(distance * DISTANCE_RATIO)    # kilometers

					'''
					According to real life conditions the time-span between stops can be as small as it can be interpeted as time spended
					for a rest or taking a break between traveling. Such case can distort final result so I decided to eliminate such 
					observation by following command (eliminate those movements where distance is less than 50 meters)
					'''

					df_person_this_timestamp = df_person_this_timestamp[df_person_this_timestamp['Distance'] > 0.05]

					if len(df_person_this_timestamp) > 0:
						median_distance_this_timestamp = np.round(np.median(df_person_this_timestamp['Distance'].values) * 100, 3)

						# Save this record to the person's dictionary
						person_travels['Day timestamp'] = this_timestamp
						person_travels['Median distance (m)'] = median_distance_this_timestamp
						person_travels['Number of travels'] = len(df_person_this_timestamp)

						l_person.append(dict(person_travels))

			# Convert list to Pandas DataFrame
			df_person_median_distances = pd.DataFrame(l_person)

			# Calculate median distances for whole active period for this user
			median_distance_TOTAL = np.median(median_distances_USER_FULL)
			print('| User {} = {} km'.format(person, round(median_distance_TOTAL, 3)))

			# Add Person's DataFrame to the main and non-aggregated lists of DataFrames
			l.append(df_person_median_distances)
			l_median_distances.append(median_distance_TOTAL)

			# Output activity table for each user (person) if needed for debuging or any
			#print('| Activity table for {} user:'.format(person))
			#print(df_person_median_distances)

			'''
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
			BONUS: somethimes it is not easy to detect if there are any dangerous data outlies in a given
				   dataset. For this I always use ECDF method for double checking. Let's do it!
			'''
			x, y = E.ecdf(median_distances_USER_FULL)
			_ = E.plot_ecdf(x, y, C.colors_for_plotting()[person], person)
		
		# Close the plot.
		plt.show()

		return l, l_median_distances

	else:
		print('Not enough data.')
		return None


# Plot users activity by median distances in meters for each day
def plot_activity(data, median_distances, colors):
	'''
	Args:
		-- data : list of <Pandas DataFrames> that store aggregated distance data for each user per day
		-- median_distances : <list> of non-aggregated distances between two subseq. locations per user
		-- colors : custom pallete of colors for plotting purposes
	'''
	
	# Set plot size
	rcParams['figure.figsize'] = 14, 4

	# Set anomalies threshold constant
	ANOMALIES_THRESHOLD_CHANGE = 10
	ANOMALIES_THRESHOLD_METERS = 5 * 1000    # Converting Km to Meters

	# Build Pandas DataFrame that joins all elementes in a given list


	_ = plt.title('Users Activity Graph', fontsize=11, family='Arial')

	# Plotting normal lines
	for i, user_data in enumerate(data):
		
		_ = plt.plot(user_data['Day timestamp'], user_data['Median distance (m)'],
					 linewidth=1, color=colors[i], label='User {}'.format(i))

		# Checking for anomalies for this user
		if len(user_data) > 2:

			index_timestamp = user_data.columns.get_loc('Day timestamp')
			index_distance = user_data.columns.get_loc('Median distance (m)')

			for current_day in range(2, len(user_data)):

				current_distance = user_data.iat[current_day, index_distance]
				previous_distance = user_data.iat[current_day-1, index_distance]

				# Checking if current distance is anomaly in compare with previous day activity.
				# For this I use ANOMALIES THRESHOLD_CHANGE and ANOMALIES_THRESHOLD_METERS constant.
				if current_distance / previous_distance > ANOMALIES_THRESHOLD_CHANGE and \
					current_distance > ANOMALIES_THRESHOLD_METERS:
					_ = plt.plot(user_data.iat[current_day, index_timestamp], user_data.iat[current_day, index_distance],
								 linewidth=0,
								 marker='D',
								 markersize=5,
								 color='red')

		# Plotting median distance value for each user
		_ = plt.axhline(y=median_distances[i] * 1000,    # Converting kilometers to meters
						color=colors[i],
						linestyle='--', linewidth=2,
						label='Median distance (User {})'.format(i))

	_ = plt.grid(which='major', color='#cccccc', linestyle='--', alpha=0.5)
	_ = plt.xlabel('Timestamp', fontsize=11, family='Arial')
	_ = plt.ylabel('Median distance \nbetween two locations per day (m)', fontsize=11, family='Arial')
	_ = plt.legend(shadow=True)
	_ = plt.ylim((0, np.max(pd.concat(data)['Median distance (m)'])+2500))
	_ = plt.show()

	return None


# Find meaningless points of locations and mark it!
def clean_map(data):
	'''
	Args:
		-- data : full Pandas DataFrame with parsed data	
	'''

	print('{} CLEANING ACTIVITY MAPS '.format(SEPARATOR))

	n = len(data['Person'].unique())

	if n > 0:

		# Extra column to mark location points
		data['Valid'] = None

		# Set indexes for columns we will need
		index_valid = data.colums.get_loc('Valid')
		index_lat = data.columns.get_loc('latitude')
		index_lng = data.columns.get_loc('longitude')

		for this_person in range(0, n):
			temp_df = data[data['Person'] == n]

			if len(temp_df) > 1:

				# Cleaning a separate part of dataframe for a single person
				for this_location in range(0, len(temp-df)):

					# List of distances between this_location on k location
					dists = [] 

					# Extract coordinates for 2 points
					x1 = data.iat[this_location, index_lat]
					y1 = data.iat[this_location, index_lng]

	else:
		print('Erorr: issue with persons.')

	return None