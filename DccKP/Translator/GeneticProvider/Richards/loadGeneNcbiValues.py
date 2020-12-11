# imports
import pandas as pd 
import pymysql as mdb

# file location
file_loc = "/home/javaprog/Data/Broad/dig-analysis-data/bin/magma/NCBI37.3.gene.loc"

# load file into pandas dataframe
gene_df = pd.read_csv (file_loc, sep = '\t', names=['ncbi_id', 'chrom', 'start', 'end', 'add', 'gene_name'])

# describe the dataframe
print("data: \n{}\n\n".format(gene_df.head(5)))

gene = 'PPARG'
print("gene {} has row \n{}".format(gene, gene_df[gene_df.gene_name == gene]))

# create connection
conn = mdb.connect(host='localhost', user='root', password='this aint no password', charset='utf8', db='richards_gene')
cur = conn.cursor()

sql = """insert into `gene_ncbi` (ncbi_id, gene)
         values (%s, %s) 
    """

# loop through rows
for index, row in gene_df.iterrows():
    ncbi_id = 'NCBIGene:' + str(row['ncbi_id'])
    print("gene {} with id {}".format(ncbi_id, row['gene_name']))

    cur.execute(sql,(ncbi_id, row['gene_name']))
conn.commit()


