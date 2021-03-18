
# imports
import pandas as pd 
import sklearn 
import time
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from category_encoders import BinaryEncoder 

print("got pandas version {}".format(pd.__version__))
print("got sklearn version {}".format(sklearn.__version__))

# debug class
class DebugModel(BaseEstimator, TransformerMixin):
    def transform(self, X):
        return X

    def fit(self, X, y=None, **fit_params):
        print("features shape {}".format(X.shape))
        print("{}".format(X.head(10)))

class PipBinaryEncoder(BaseEstimator, TransformerMixin):
    def transform(self, X):
        return X

    def fit(self, X, y=None, **fit_params):
        print("features shape {}".format(X.shape))
        print("{}".format(X.head(10)))


# constants
timestr = time.strftime("%Y%m%d-%H%M%S")
home_dir = "/home/javaprog/Data/Personal"
home_dir = "/Users/mduby/Data"
train_file = home_dir + "/Kaggle/202103tabularPlayground/train.csv"
test_file = home_dir + "/Kaggle/202103tabularPlayground/test.csv"
submission_file = home_dir + "/Kaggle/202103tabularPlayground/Submissions/" + timestr + "-{}-{}-submit.csv"
random_state = 23

# read the data
df = pd.read_csv(train_file)
print("features shape {}".format(df.shape))
print("{}".format(df.head(10)))

# get the column names
columns = list(df.columns)
print("got data columns {}".format(columns))

# loop through columns, get unique values
for col in columns:
    if 'cat' in col:
        unique = df[col].unique()
        # print("col {} got {} values: {}".format(col, len(unique), unique))
        print("col {} got {} values".format(col, len(unique)))

# categorical columns
categorical = [cat for cat in columns if 'cat' in cat]
print("categorical columns {}".format(categorical))

# get the features/labels
X = df.drop(['id', 'target'], axis=1)
y = df['target']

# create the pipeline and transform
pipeline = Pipeline([
    ('binary_enc', BinaryEncoder(cols=categorical)),
    ('debug', DebugModel())
])
pipeline.fit(X, y)
