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

if __name__ == "__main__":
    # load test data
    home_dir = "/home/javaprog/Data/Personal"
    home_dir = "/Users/mduby/Data"
    train_file = home_dir + "/Kaggle/202103tabularPlayground/train.csv"
    df = pd.read_csv(train_file)

    # upsample
    df_up = resample_dataset(df)

    # downsample
    df_down = resample_dataset(df, up=False)


