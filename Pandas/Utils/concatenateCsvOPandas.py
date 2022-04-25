
# import
import pandas as pd 
import csv
import glob
import os

# directory locations
dir_input = "/home/javaprog/Data/Broad/Scratch/TestPandas/BMI/ancestry=AF"
dir_output = "/home/javaprog/Data/Broad/Scratch/TestPandas/BMI"

# main program
# load the AF csv files
all_files = glob.glob(os.path.join(f'{dir_input}', "part*.csv"))

# concatenate them
li = []
test = False

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, sep="\t")
    if not test:
        print(df.head(n=5))
        test = True
    li.append(df)
    print("read {} rows for input file: {}".format(len(df), filename))

df_input = pd.concat(li, axis=0, ignore_index=True)

# save the file back to disk
file_output = "{}/BMI_AF.csv".format(dir_output)
df_input.to_csv(file_output, index=False, sep="\t")
print("\n\nwrote out {} rows to file: {}".format(len(df_input), filename))

print(df_input.head(5))




# df_input = pd.concat((pd.read_csv(file_temp) for file_temp in all_files))
# file_input = f'{dir_cojo}/input/cojo_{phenotype}_{ancestry}.csv'



