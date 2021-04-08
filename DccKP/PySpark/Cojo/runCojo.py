# imports
import pandas as pd 
import boto3
import argparse
import re
import os
import glob 
import time 

# map for the cojo ancestry swithes
map_ancestry = {'EU': 'eur', 'EA': 'eas', 'AA': 'afr', 'HS': 'amr', 'SA': 'sas'}


def run_system_command(os_command, input_message = "", if_test = True):
    ''' method to run an OS command and time it'''
    log_message = "Running command"
    exit_code = None
    start = time.time()
    if if_test:
        log_message = "Testing command"
    print("{}: {}".format(log_message, os_command))
    if not if_test:
        exit_code = os.system(os_command)
    end = time.time()
    print("    Done in {:0.2f}s with exit code {}".format(end - start, exit_code))

def main():
    """
    Arguments: phenotype
    """
    opts = argparse.ArgumentParser()
    opts.add_argument('phenotype')

    # parse command line
    args = opts.parse_args()
    phenotype = args.phenotype 

    # constants
    arg_if_test=False
    dir_s3 = f'dig-analysis-data'
    dir_cojo = f'/home/javaprog/Temp/Cojo'
    # dir_cojo = f'/home/javaprog/Temp/Cojo'
    dir_cojo_out = f'{dir_cojo}/output'
    dir_cojo_in = f'{dir_cojo}/input'
    dir_s3_inputs = f'out/finemapping/variant-associations/{phenotype}'
    dir_s3_outputs = f'{dir_s3}/out/finemapping/cojo-results/{phenotype}'


    # read in the ancestries in the phenotype
    client = boto3.client('s3')
    ancestries = []
    result = client.list_objects(Bucket=dir_s3, Prefix=f'{dir_s3_inputs}/', Delimiter='/')
    for folder in result.get('CommonPrefixes'):
        # match = re.search(pattern='/ancestry=([^/]+)/', string="{}".format(folder))
        match = re.search(pattern='stry=([A-W]*)', string="{}".format(folder))
        ancestry = match.group()
        if ancestry:
            ancestry = ancestry[-2:]
        print("sub folder : {} and ancestry {}".format(folder.get('Prefix'), ancestry))
        ancestries.append(ancestry)

    # for each ancestry
    for ancestry in ancestries:
        if ancestry in map_ancestry.keys():
            # copy files
            dir_s3_ancestry = f'{dir_s3_inputs}/ancestry={ancestry}'
            s3_command = f'aws s3 cp --recursive --include "part*.csv" s3://{dir_s3}/{dir_s3_ancestry}/ {dir_cojo}/input'
            run_system_command(s3_command, if_test = arg_if_test)

            # combine files and write out
            all_files = glob.glob(os.path.join(f'{dir_cojo}/input', "part*.csv"))
            df_input = pd.concat((pd.read_csv(file_temp) for file_temp in all_files))
            file_input = f'{dir_cojo}/input/cojo_{phenotype}_{ancestry}.csv'
            df_input.to_csv(file_input, index=False)

            for chromosome in range(1, 23):
                # run cojo command with appropriate ancestry g1000 files
                file_g1000 = f'{dir_cojo}/g1000_{map_ancestry.get(ancestry)}'
                file_output = f'{dir_cojo}/output/out_{phenotype}_{ancestry}_{chromosome}'
                # cojo_command = f'{dir_cojo}/gcta_1.93.2beta/gcta64 --bfile {file_g1000} --maf 0.01 --cojo-file {file_input} --cojo-wind 500 --threads 8 --cojo-slct --out {file_output}'
                cojo_command = f'{dir_cojo}/gcta_1.93.2beta/gcta64 --bfile {file_g1000} --maf 0.005 --chr {chromosome} --cojo-file {file_input} --cojo-wind 500 --threads 10 --cojo-slct --out {file_output}'
                run_system_command(cojo_command, if_test = arg_if_test)

                # copy results to s3 new directory
                s3_upload_command = f'aws s3 cp --recursive --exclude "*" --include "out_{phenotype}_{ancestry}_{chromosome}*.*" {dir_cojo}/output/ s3://{dir_s3_outputs}/ancestry={ancestry}/'
                run_system_command(s3_upload_command, if_test = arg_if_test)

            # cleanup the input and output drectory
            fs_input_command = f'rm -rf {dir_cojo}/input/*.csv'
            fs_output_command = f'rm -rf {dir_cojo}/output/*.*'
            run_system_command(fs_input_command, if_test = arg_if_test)
            run_system_command(fs_output_command, if_test = arg_if_test)
 
    # log
    print("phenotype {} run through cojo for ancestries {}".format(phenotype, ancestries))

if __name__ == "__main__":
    main()
