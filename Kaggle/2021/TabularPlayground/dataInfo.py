
# imports
import pandas as pd 
import sklearn 

print("got pandas version {}".format(pd.__version__))
print("got sklearn version {}".format(sklearn.__version__))

# constants
train_file = "/home/javaprog/Data/Personal/Kaggle/202103tabularPlayground/train.csv"

# read the data
df = pd.read_csv(train_file)
df.info()


# get the column names
columns = list(df.columns)
print("got data columns {}".format(columns))

# loop through columns, get unique values
for col in columns:
    if 'cat' in col:
        print("col {} got values: {}".format(col, df[col].unique()))
