
# imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# build the data
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0.1, 0.2, 0.5, 0.6, 0.8, 1.2, 1.0, 1.2]

# print
print(x)

# plot
# plt.plot(x, y)
plt.bar(x, y)

# log
print('plot created')

# set options
# plt.interactive(False)

# show the plot
plt.show()
print('show plot')


