# imports
import pandas as pd 
from sklearn.utils import resample


def resample_dataset(df, up=True, random_state=23, log=True):
    ''' method to rebalance a binary dataset '''
    # find out the balance
    if log:
        print("input dataset of shape {}".format(df.shape))
    counts = df['target'].value_counts()
    new_size = None
    target_min = counts.idxmin()
    target_max = counts.idxmax()

    # get the highest/lowest
    if up:
        new_size = counts.max()
    else:
        new_size = counts.min()

    # split the dataset
    df_min = df[df.target == target_min]
    df_max = df[df.target == target_max]
    if log:
        print("got min df of size {} and target {} with max df of size {} and target {}".format(df_min.shape, target_min, df_max.shape, target_max))

    # resample
    if up:
        old_shape = df_min.shape
        df_min = resample(df_min,
            replace=True,
            n_samples=new_size,
            random_state=random_state)
        if log:
            print("upsample min from {} to {}".format(old_shape, df_min.shape))
            
    else:
        old_shape = df_max.shape
        df_max = resample(df_max,
            replace=False,
            n_samples=new_size,
            random_state=random_state)
        if log:
            print("downsample max from {} to {}".format(old_shape, df_max.shape))

    # recombine the data
    df_return = pd.concat([df_min, df_max], axis=0)

    # return
    if log:
        print("resampled dataset of shape {}".format(df_return.shape))
    return df_return

def pseudo_sample_fit(model, X_train, y_train, X_test):
    ''' method to pseudo label test data and retrain network with combined dataset '''
    # train the model
    model.fit(X_train, y_train)

    # predict on test data
    y_pred = model.predict(X_test)

    # combine train and test datasets
    X_combined = pd.concat([X_train, X_test], axis=0)
    y_combined = pd.concat([y_train, pd.Series(y_pred, name='target')], axis=0)

    # retrain the model
    model.fit(X_combined, y_combined)

    # return the model
    return model

def one_hot_dummies(X_train, X_test, categorical_columns, log=True):
    ''' combines the train and test DF and one hots the combined df '''
    # add column to dataframes
    if log:
        print("got train dataset {} and test dataset {}".format(X_train.shape, X_test.shape))
    X_train['split'] = 'train'
    X_test['split'] = 'test'

    # combine data frames
    X_combined = pd.concat([X_train, X_test], axis=0)

    # one hot
    X_combined = pd.get_dummies(X_combined, columns=categorical_columns)

    # split data frames
    X_rtrain = X_combined[X_combined['split'] == 'train']
    X_rtest = X_combined[X_combined['split'] == 'test']

    # drop extra column
    X_rtrain = X_rtrain.drop(['split'], axis=1)
    X_rtest = X_rtest.drop(['split'], axis=1)

    # return
    if log:
        print("after dummies, got train dataset {} and test dataset {}".format(X_rtrain.shape, X_rtest.shape))
    return X_rtrain, X_rtest

if __name__ == "__main__":
    # load test data
    home_dir = "/home/javaprog/Data/Personal"
    home_dir = "/Users/mduby/Data"
    train_file = home_dir + "/Kaggle/202103tabularPlayground/train.csv"
    test_file = home_dir + "/Kaggle/202103tabularPlayground/test.csv"
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # upsample
    df_up = resample_dataset(df_train)

    # downsample
    df_down = resample_dataset(df_train, up=False)

    # one hot encode the data
    columns = df_train.columns
    categorical = [cat for cat in columns if 'cat' in cat]
    df_one_hot = one_hot_dummies(df_train.drop(['target'], axis=1), df_test, categorical)


