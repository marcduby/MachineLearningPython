
# imports
import pandas as pd

# constants
file_results = "/home/javaprog/Data/Personal/RaceResults/2021hocr/2021hocr.txt"
file_gmaster = "/home/javaprog/Data/Personal/RaceResults/2021hocr/2021gmaster.tsv"

# load the data
with open(file_results, 'r') as handle_file:
    # create the dict array
    array_dict = []

    # loop
    for i in range(122):
        # initialize
        entry = []

        # read place and bow
        item = handle_file.readline().rstrip()
        entry = entry + item.split("\t")

        # skip icon line
        item = handle_file.readline()

        # read org
        item = handle_file.readline().rstrip()
        entry = entry + item.split("\t")

        # read name, division, time1, time2
        item = handle_file.readline().rstrip()
        entry = entry + item.split("\t")

        # read split2, time3
        item = handle_file.readline().rstrip()
        entry = entry + item.split("\t")

        # read split3, time4
        item = handle_file.readline().rstrip()
        entry = entry + item.split("\t")

        # read split4, percentage, time from winner
        item = handle_file.readline().rstrip()
        entry = entry + item.split("\t")

        # print
        print(entry)

        # create the dict
        dict_entry = {'place': entry[0], 'bow': entry[1], 'club': entry[2], 'name': entry[3], 'event': entry[4], 'time': entry[10]}
        array_dict.append(dict_entry)

# parse into dataframe
# create the dataframe
df_results = pd.DataFrame(array_dict)
print(df_results.info())
print(df_results.head(10))

df_gmaster = df_results[df_results['event'] == 'Grand Master 1x']
df_gmaster = df_gmaster.reset_index()
print(df_gmaster.head(100))

# save data
df_gmaster.to_csv(file_gmaster, sep='\t')

