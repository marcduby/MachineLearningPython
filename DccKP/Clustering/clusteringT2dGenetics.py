# imports
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import numpy as np 

print("got pandas version {}".format(pd.__version__))


# file locations
file_location = "/home/javaprog/Data/Broad/T2dCluster/t2d_loci_traits120617.txt"
row_number = 10
random_seed = 23
number_estimators = 100

# read the file
file_df = pd.read_csv(file_location, sep='\t')
file_df.bin = pd.Categorical(file_df.bin)
file_df['bin_num'] = file_df.bin.cat.codes
file_df.info()

# print the first 10 rows
print("the file data: \n{}".format(file_df.head(row_number)))

# get the target and features
X = file_df.drop(labels=['snpusedbyme', 'gene', 'bin'], axis='columns')
y = X.pop('bin_num')
print("got features with shape {} and targets with shape {}".format(X.shape, y.shape))
# X.info()
feature_names = X.columns
print("have unique targets: {}".format(y.unique()))

# clean the data
X = X.fillna(X.mean())
X.info()

# test to make sure all features are good
# print("are there any bad inputs: {}".format(np.isnan(X)))

# build the model
model = RandomForestClassifier(n_estimators=number_estimators, random_state=random_seed)
model.fit(X, y)

# get the feature importance
feature_importances = model.feature_importances_
importance = sorted(zip(feature_importances, feature_names), reverse=True)
print("features ranked by importance:")
[print("{}: {}".format(i, row)) for i, row in enumerate(importance)]



