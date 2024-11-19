

import pandas as pd
import json
import csv 
import copy 

# constamts
FILE_XLX = "/Users/mduby/Data/Broad/Hackathon/Distance/pan_gene_missense_mutations_to_CKAY_PDB_raw_empty_removed_smallset_merged.xlsx"

FILE_CSV = "/Users/mduby/Data/Broad/Hackathon/Distance/min_distance_from_{}_to_{}.csv"

FILE_RESULT = "/Users/mduby/Data/Broad/Hackathon/Distance/mergedResidue.tsv"

def write_tab_delimited_file(json_list, output_file, column_order=None):
    list_to_print = []

    # Check if the list is empty
    if not json_list:
        raise ValueError("The provided JSON list is empty")

    # If no column order is provided, use the keys from the first dictionary
    if column_order is None:
        column_order = json_list[0].keys()
    
    # build new list with only columns needed
    for item in json_list:
        map_temp = {}
        for col in column_order:
            map_temp[col] = item.get(col)
        list_to_print.append(map_temp)

    # Open the output file in write mode
    with open(output_file, 'w', newline='') as file:
        # Create a CSV writer object with tab delimiter
        writer = csv.DictWriter(file, fieldnames=column_order, delimiter='\t')

        # Write the header
        writer.writeheader()

        # Write the data
        # for item in json_list:
        for item in list_to_print:
            writer.writerow(item)



def read_csv(file_name):
    print("reading csv file: {}".format(file_name))

    with open(file_name, mode='r') as file:
        # Create a CSV DictReader
        csv_reader = csv.DictReader(file)
    
        # Convert the CSV data into a list of dictionaries
        list_data = [row for row in csv_reader]

    return list_data


if __name__ == "__main__":
    # Read the Excel file
    df = pd.read_excel(FILE_XLX)
    list_final_rows = []

    # Convert the DataFrame to a list of dictionaries
    list_of_dicts = df.to_dict(orient='records')

    # Display the list of dictionaries
    print(json.dumps(list_of_dicts, indent=2))

    # loop through rows
    for row in list_of_dicts:
        # read in the file
        file_name = FILE_CSV.format(row.get('Entry ID'), row.get('Ligand ID'))

        list_csv_rows = read_csv(file_name=file_name)
        print(json.dumps(list_csv_rows, indent=2))

        for rowa in list_csv_rows:
            # copy the master map
            map_copy = copy.deepcopy(row)

            # add in the residue
            map_copy['Residue_Number'] = rowa.get('Residue_Number')

            # add to list
            list_final_rows.append(map_copy)

    # save file
    write_tab_delimited_file(output_file=FILE_RESULT, json_list=list_final_rows)

