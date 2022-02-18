
# imports
import pandas as pd 
import pymysql as mdb

# define fiel path
dir_load = "/home/javaprog/Data/Broad/Lipids/Load/"
file_u20s = dir_load + 'LD_protein_in_U20S_cells.csv'
file_huh7 = dir_load + 'LD_protein_in_Huh7_cells.csv'
file_rna = dir_load + 'RNAi_screen_hit_pairwise_correlations.csv'

map_files = {
    'u20s': [file_u20s, 'lipids_ld_protein_u20s'],
    'huh7': [file_huh7, 'lipids_ld_protein_huh7'],
    'rnai': [file_rna, 'lipids_ld_protein_rnai']
}

# create connection
conn = mdb.connect(host='localhost', user='root', password='yoyoma', charset='utf8', db='lipids_load')
cur = conn.cursor()

# load the file
for key, values in map_files.items():
    print("\n\nfor {}".format(key))
    df_data = pd.read_csv(values[0])

    # inspect
    df_data.info()
    print(df_data.head(5))

    # delete from the tables
    sql_delete = "delete from {}".format(values[1])
    print("data deleted")

    # load the data
    if key == 'rnai':
        i = 0
        sql_insert = "insert into {} (gene_source, gene_target, rho_value) values(%s, %s, %s)".format(values[1])
        for index, row in df_data.iterrows():
            source = str(row['source']).strip()
            target = str(row['target']).strip()
            i += 1
            if i % 10000 == 0:
                print("gene source {} with target {}".format(source, target))

            cur.execute(sql_insert,(source, target, row['rho-value']))
        conn.commit()

    else:
        i = 0
        sql_insert = "insert into {} (gene, csn_value) values(%s, %s)".format(values[1])
        for index, row in df_data.iterrows():
            gene = str(row['Symbol']).strip()
            i += 1
            if i % 100 == 0:
                print("gene source {} with csn value {}".format(gene, row['CSN']))

            cur.execute(sql_insert,(gene, row['CSN']))
        conn.commit()

# close the connection
cur.close()
conn.close()


# look at data
# df_clingen = df_clingen[['GENE SYMBOL', 'GENE ID (HGNC)', 'DISEASE LABEL', 'DISEASE ID (MONDO)', 'CLASSIFICATION', 'GCEP']]
# print("got data {}\n".format(df_clingen.head(20)))

# # load the file
# df_clinvar = pd.read_csv(file_clinvar, sep='\t')

# # inspect
# df_clinvar.info()

# # look at data
# # df_clinvar = df_clinvar.loc[df_clinvar['SourceName'] != 'NaN']
# # df_clinvar = df_clinvar[df_clinvar['SourceID'].notna()]
# df_clinvar = df_clinvar[df_clinvar['DiseaseName'].notna()]
# print("\n\ngot data {}\n{}\n".format(df_clinvar.shape[0], df_clinvar.head(20)))

# # unique values for classification
# print("\nthe unique classification for clingen is")
# for classinfication in df_clingen['CLASSIFICATION'].unique():
#     print(classinfication)


# # create connection
# conn = mdb.connect(host='localhost', user='root', password='yoyoma', charset='utf8', db='tran_dataload')
# cur = conn.cursor()


# # load the clingen data
# sql = """insert into `clingen_gene_phenotype` (gene, gene_id, phenotype, phenotype_id, provenance, classification)
#          values (%s, %s, %s, %s, %s, %s) 
#     """
# i = 0
# # loop through rows
# for index, row in df_clingen.iterrows():
#     i += 1
#     gene_id = str(row['GENE SYMBOL'])
#     if i % 200 == 0:
#         print("gene {} with disease {}".format(gene_id, row['DISEASE LABEL']))

#     cur.execute(sql,(gene_id, row['GENE ID (HGNC)'], row['DISEASE LABEL'], row['DISEASE ID (MONDO)'], 'Clingen', row['CLASSIFICATION']))
# conn.commit()

# # split the clinvar data
# df_clinvar = df_clinvar.where(pd.notnull(df_clinvar), None)
# df_clinvar_associated = df_clinvar[df_clinvar['AssociatedGenes'].notna()]
# df_clinvar_related = df_clinvar[df_clinvar['RelatedGenes'].notna()]
# print("got associated {} and related {}".format(df_clinvar_associated.shape[0], df_clinvar_related.shape[0]))

# # load the clinvar data
# sql = """insert into `clingen_gene_phenotype` (gene, phenotype, phenotype_id, provenance, classification)
#          values (%s, %s, %s, %s, %s) 
#     """

# # loop through rows
# for index, row in df_clinvar_associated.iterrows():
#     i += 1
#     gene_id = str(row['AssociatedGenes'])
#     if i % 200 == 0:
#         print("gene {} with disease {}".format(gene_id, row['DiseaseName']))

#     cur.execute(sql,(gene_id, row['DiseaseName'], row['SourceID'], 'Clinvar', 'Associated'))

# conn.commit()

# for index, row in df_clinvar_related.iterrows():
#     i += 1
#     gene_id = str(row['RelatedGenes'])
#     if i % 200 == 0:
#         print("gene {} with disease {}".format(gene_id, row['DiseaseName']))

#     cur.execute(sql,(gene_id, row['DiseaseName'], row['SourceID'], 'Clinvar', 'Related'))

# conn.commit()

