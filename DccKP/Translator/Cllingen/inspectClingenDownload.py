
# imports
import pandas as pd 
import pymysql as mdb

# define fiel path
file_clingen = '/home/javaprog/Data/Broad/Translator/Clingen/Clingen-Gene-Disease-Summary-2021-05-04.formatted.csv'
file_clinvar = '/home/javaprog/Data/Broad/Translator/Clingen/clinvar.gene_condition_source_id.tsv'

# load the file
df_clingen = pd.read_csv(file_clingen)

# inspect
df_clingen.info()

# look at data
df_clingen = df_clingen[['GENE SYMBOL', 'GENE ID (HGNC)', 'DISEASE LABEL', 'DISEASE ID (MONDO)', 'CLASSIFICATION', 'GCEP']]
print("got data {}\n".format(df_clingen.head(20)))

# load the file
df_clinvar = pd.read_csv(file_clinvar, sep='\t')

# inspect
df_clinvar.info()

# look at data
# df_clinvar = df_clinvar.loc[df_clinvar['SourceName'] != 'NaN']
# df_clinvar = df_clinvar[df_clinvar['SourceID'].notna()]
df_clinvar = df_clinvar[df_clinvar['DiseaseName'].notna()]
print("\n\ngot data {}\n{}\n".format(df_clinvar.shape[0], df_clinvar.head(20)))

# unique values for classification
print("\nthe unique classification for clingen is")
for classinfication in df_clingen['CLASSIFICATION'].unique():
    print(classinfication)


# create connection
conn = mdb.connect(host='localhost', user='root', password='yoyoma', charset='utf8', db='tran_dataload')
cur = conn.cursor()


# load the clingen data
sql = """insert into `clingen_gene_phenotype` (gene, gene_id, phenotype, phenotype_id, provenance, classification)
         values (%s, %s, %s, %s, %s, %s) 
    """
i = 0
# loop through rows
for index, row in df_clingen.iterrows():
    i += 1
    gene_id = str(row['GENE SYMBOL'])
    if i % 200 == 0:
        print("gene {} with disease {}".format(gene_id, row['DISEASE LABEL']))

    cur.execute(sql,(gene_id, row['GENE ID (HGNC)'], row['DISEASE LABEL'], row['DISEASE ID (MONDO)'], 'Clingen', row['CLASSIFICATION']))
conn.commit()

# split the clinvar data
df_clinvar = df_clinvar.where(pd.notnull(df_clinvar), None)
df_clinvar_associated = df_clinvar[df_clinvar['AssociatedGenes'].notna()]
df_clinvar_related = df_clinvar[df_clinvar['RelatedGenes'].notna()]
print("got associated {} and related {}".format(df_clinvar_associated.shape[0], df_clinvar_related.shape[0]))

# load the clinvar data
sql = """insert into `clingen_gene_phenotype` (gene, phenotype, phenotype_id, provenance, classification)
         values (%s, %s, %s, %s, %s) 
    """

# loop through rows
for index, row in df_clinvar_associated.iterrows():
    i += 1
    gene_id = str(row['AssociatedGenes'])
    if i % 200 == 0:
        print("gene {} with disease {}".format(gene_id, row['DiseaseName']))

    cur.execute(sql,(gene_id, row['DiseaseName'], row['SourceID'], 'Clinvar', 'Associated'))

conn.commit()

for index, row in df_clinvar_related.iterrows():
    i += 1
    gene_id = str(row['RelatedGenes'])
    if i % 200 == 0:
        print("gene {} with disease {}".format(gene_id, row['DiseaseName']))

    cur.execute(sql,(gene_id, row['DiseaseName'], row['SourceID'], 'Clinvar', 'Related'))

conn.commit()

