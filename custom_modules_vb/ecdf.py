'''
This module is responsible to build ECDF (A cummulative distribution function) for
a given series of data. This method is useful to detect data outliers for the task.

Prepared by Vytautas Bielinskas
'''

# Import modules and packages
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 5, 5

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def ecdf(data):
    """
    Compute ECDF for a one-dimensional array of measurements.

	Args:
		data = a vector of values have to be ploted as ECD function.

    """

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, len(x)+1) / n

    return x, y


# Plot ECDF graph
def plot_ecdf(x, y, user_color, user):
	'''
		Args: x, y : vectors of generated values from a dataset for ECDF.

		(!!!) A plot is closing outside this module.
	'''

	_ = plt.title('ECDF of median values of distances for each user', family='Arial', fontsize=11)

	_ = plt.plot(x, y,
				 marker = '.',
				 markersize=2,
				 color = user_color,
				 label = 'User {}'.format(user))

	_ = plt.grid(which='major',
		color='#cccccc',
		alpha=0.5,
		linestyle='--')

	_ = plt.xlabel('Median distance (meters) between two subs. locations', family='Arial', fontsize=9)
	_ = plt.ylabel('Percent of all range, % ', family='Arial', fontsize=9)

	_ = plt.legend(shadow=True, loc='best')

	return _