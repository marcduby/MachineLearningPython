

# imports
import pandas as pd 


# constants
location_data = "/Users/mduby/Data/Broad/"
file_eve_pparg = location_data + "EveEvolutionaryML/Analyze/PPARG_HUMAN.csv"
file_miter_pparg = location_data + "EveEvolutionaryML/Analyze/matrixHeat.csv"

# read the csv
df_eve = pd.read_csv(file_eve_pparg)
df_miter = pd.read_csv(file_miter_pparg)

# subselect columns
df_eve_smaller = df_eve[['position', 'mt_aa', 'EVE_scores_ASM']]

# pivot
df_eve_pivot = df_eve_smaller.pivot(index='position', columns='mt_aa')

# analyze
print("got eve data \n{}".format(df_eve_pivot.tail(20)))

df_miter = df_miter.set_index('Pos')
print("got miter data \n{}".format(df_miter.tail(20)))


