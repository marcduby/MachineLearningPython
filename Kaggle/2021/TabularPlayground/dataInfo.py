
# imports
import pandas as pd 
import sklearn 

print("got pandas version {}".format(pd.__version__))
print("got sklearn version {}".format(sklearn.__version__))

# constants
train_file = "/home/javaprog/Data/Personal/Kaggle/202103tabularPlayground/train.csv"
train_file = "/Users/mduby/Data/Kaggle/202103tabularPlayground/train.csv"
test_file = "/Users/mduby/Data/Kaggle/202103tabularPlayground/test.csv"

# read the data
df = pd.read_csv(train_file)
df.info()


# get the column names
columns = list(df.columns)
print("got data columns {}".format(columns))

# loop through columns, get unique values
for col in columns:
    if 'cat' in col:
        unique = df[col].unique()
        print("col {} got {} values: {}".format(col, len(unique), unique))


# df = pd.get_dummies(df, prefix=columns)
# df = pd.get_dummies(df, prefix=['cat12'])
X = df.drop(['id', 'target'], axis=1)
y = df['target']
X = pd.get_dummies(X)
X.info()
