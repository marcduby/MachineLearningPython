
# imports
from torchvision.datasets.utils import download_and_extract_archive
from pathlib import Path
import pandas as pd 

print("pandas verwsion of {}".format(pd.__version__))


# set the data constants
FER_URL = "https://www.dropbox.com/s/2rehtpc6b5mj9y3/fer2013.tar.gz?dl=1"
DOWNLOAD_DIR = "/home/javaprog/Data/Personal/Courses/Numfocus/Pytorch/"
DOWNLOAD_ROOT = Path(DOWNLOAD_DIR)
FILENAME = "fer2013.tar.gz"
MD5 = "ca95d94fe42f6ce65aaae694d18c628a"

# download the data
download_and_extract_archive(url=FER_URL, download_root = DOWNLOAD_ROOT, filename= FILENAME, md5= MD5)

# load into pandas
df_per = pd.read_csv(DOWNLOAD_DIR + "/fer2013/fer2013.csv")
print("the shape of the data is {}".format(df_per.shape))
print("head: \n{}\n".format(df_per.head(5)))
print("columns: \n{}\n".format(df_per.columns))

# describe the dataframe
df_per.info()

# look at the label columns
print("the values for Usage are {}".format(df_per.Usage.unique()))
print("the value counts for Usage are \n{}".format(df_per.Usage.value_counts()))

print("the values for emotion are {}".format(df_per.emotion.unique()))
print("the value counts for emotion are \n{}".format(df_per.emotion.value_counts()))

# categortize the labels
df_per.emotion = pd.Categorical(df_per.emotion)
print("the unique emotion codes are {}\n".format(df_per.emotion.cat.codes.unique()))
df_per.info()

# add a column for explainability of labels
map_emotions = {0: 'angry', 1: 'digust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
df_per["emotion_str"] = df_per.emotion.apply(lambda x: map_emotions[x])
print("the update data frame is: \n{}".format(df_per.head(10)))


