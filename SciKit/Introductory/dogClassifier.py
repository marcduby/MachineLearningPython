# imports
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as py_plot

# build data
greyhounds_num = 500
labs_num = 500

# build haeight distributions
grehound_height = 28 + (4 * np.random.randn(greyhounds_num))
labs_height = 24 + (4 * np.random.randn(labs_num))

# polt
py_plot.hist([grehound_height, labs_height], stacked = True, color = ['r', 'g'])
py_plot.show()

