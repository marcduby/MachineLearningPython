
# look at 
# http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=1&bgcolor=FFFEE8
# http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=2&bgcolor=FFFEE8
# http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=3&bgcolor=FFFEE8


# imports
import pandas as pd


# constants
file_labels_lookup = "/home/javaprog/Data/OldNUC/Broad/Basset/Production/bassetLabelLookup.tsv"
file_bassett_labels = "/home/javaprog/Data/OldNUC/Broad/Basset/Production/basset_labels.txt"

# read the data file
df_labelnames = pd.read_csv(file_labels_lookup, sep='\t')

# print
print(df_labelnames.head(10))
print()
print(df_labelnames.info())

# read the data file
df_bassett_labels = pd.read_csv(file_bassett_labels, names=['cell'])

# print
print()
print(df_bassett_labels.info())
print()
print(df_bassett_labels.head(10))

# inner join the two cell label data 
df_joined = pd.merge(df_bassett_labels, df_labelnames, on='cell')
print()
print(df_joined.info())
print()
print(df_joined.head(10))

