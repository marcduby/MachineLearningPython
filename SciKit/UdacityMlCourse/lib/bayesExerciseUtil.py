
# imports
import random

# make the terrain data
def makeDataExample01(n_points = 1000, test_split = 0.75):
    # seed the random number
    random.seed(11)

    # create lists needed
    grade = [random.random() for i in range(0, n_points)]
    terrain = [random.random() for i in range(0, n_points)]
    error = [random.random() for i in range(0, n_points)]

    # create the y data
    y = [round(grade[i] * terrain[i] + 0.3 + 0.1 * error[i]) for i in range(0, n_points)]
    for i in range(0, len(y)):
        if grade[i] > 0.8 or terrain[i] > 0.8:
            y[i] = 1.0

    # split into train and test data
    X = [[gg, ss] for gg, ss in zip(grade, terrain)]
    split = int(test_split * n_points)
    X_train = X[0: split]
    X_test = X[split:]
    y_train = y[0: split]
    y_test = y[split:]

    # return
    return X_train, X_test, y_train, y_test


def get_plot_data(features, labels):
    first_blue = [features[i][0] for i in range(0, len(features)) if labels[i] == 0]
    second_blue = [features[i][1] for i in range(0, len(features)) if labels[i] == 0]
    first_red = [features[i][0] for i in range(0, len(features)) if labels[i] == 1]
    second_red = [features[i][1] for i in range(0, len(features)) if labels[i] == 1]

    # return
    return first_blue, second_blue, first_red, second_red


# test block
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = makeDataExample01(10, 0.8)
    print("Train X is {} and test X is{}".format(X_train, X_test))
    print("Train y is {} and test y is{}".format(y_train, y_test))

    f1, f2, d1, d2 = get_plot_data(X_train, y_train)
    print("f1 is {} and f2 is {}".format(f1, f2))


