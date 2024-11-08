# notes
# to generate ncbi file: 
#       mysql -u root -p tran_genepro -e "select * from gene_lookup" -B > gene.tsv
# to generate efo/mondo file: 
#       mysql -u root -p tran_genepro -e "select * from phenotype_id_lookup" -B > phenotype_efo_mondo.tsv
# 
# to load the generated file
#       SET GLOBAL local_infile=1;
#       mysqlimport --ignore-lines=1 --fields-terminated-by='\t' --local -u root -p tran_genepro magma_gene_phenotype.tsv
#


# imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, IntegerType
from pyspark.sql.functions import col, struct, explode, when, lit, array, udf

# load and output directory
# magma_srcdir = 's3://dig-analysis-data/out/magma/gene-associations/*/part-*'
# bin_srcdir = 's3://dig-analysis-data/bin/translator'
# outdir = 's3://dig-analysis-data/out/burdenbinning/results'

# development localhost directories
magma_srcdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/magma/gene-associations/*/part-*'
# magma_srcdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/magma/gene-test/*/part-*'
ncbi_file = '/home/javaprog/Data/Broad/dig-analysis-data/bin/translator/ncbi.tsv'
phenotype_file = '/home/javaprog/Data/Broad/dig-analysis-data/bin/translator/phenotype_efo_mondo.tsv'
bin_srcdir = '/home/javaprog/Data/Broad/dig-analysis-data/bin/translator/'
outdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/translator/results'

# constants
category_disease = 'biolink:Disease'
category_phenotype = 'biolink:PhenotypicFeature'

# print
# print("the input directory is: {}".format(vep_srcdir))


def process_magma(spark, magma_source_dir, ncbi_file):
    """
    load the magma gene data and joined with the loaded gene ncbi code file
    """
    df_magma_genes = spark.read.json(magma_source_dir)
    df_ncbi = spark.read.csv(ncbi_file, sep=r'\t', header=True).select('gene','ncbi_id')

    # print
    print("got magma row count of: {}".format(df_magma_genes.count()))

    # # join with genes for region data
    df_magma_ncbi = df_magma_genes.join(df_ncbi, on='gene', how='inner')

    # print
    print("got magma with ncbi IDs row count of: {}".format(df_magma_ncbi.count()))
    
    # # sort by gene, then by p-value
    # df.orderBy(['gene', 'pValue']) \
    #     .write \
    #     .mode('overwrite') \
    #     .json('%s/gene' % OUTDIR)

    # # sort by phenotype, then by p-value for the gene finder
    # df.orderBy(['phenotype', 'pValue']) \
    #     .write \
    #     .mode('overwrite') \
    #     .json('s3://dig-bio-index/finder/gene')

    # return
    return df_magma_ncbi

def join_magma_phenotype_ontologies(spark, df_magma_genes, phenotype_file):
    """
    join the magma gene data frame with the loaded gene efo/mondo phenotype code file
    """
    df_phenotype = spark.read.csv(phenotype_file, sep=r'\t', header=True).select('phenotype_code','tran_lookup_id','tran_lookup_name','category','group_name','dichotomous')

    # print
    print("got magma row count of: {}".format(df_magma_genes.count()))

    # fix the phenotype df for join
    df_phenotype = df_phenotype.select(
        df_phenotype.phenotype_code.alias('phenotype'),
        df_phenotype.category,
        df_phenotype.group_name,
        df_phenotype.dichotomous,
        df_phenotype.tran_lookup_id,
        df_phenotype.tran_lookup_name,
    )

    # # join with genes for region data
    df_magma_phenotype_ontology = df_magma_genes.join(df_phenotype, on='phenotype', how='inner')

    # print
    print("got magma with phenotype ontology IDs row count of: {}".format(df_magma_phenotype_ontology.count()))
    
    # # sort by gene, then by p-value
    # df.orderBy(['gene', 'pValue']) \
    #     .write \
    #     .mode('overwrite') \
    #     .json('%s/gene' % OUTDIR)

    # # sort by phenotype, then by p-value for the gene finder
    # df.orderBy(['phenotype', 'pValue']) \
    #     .write \
    #     .mode('overwrite') \
    #     .json('s3://dig-bio-index/finder/gene')

    # return
    return df_magma_phenotype_ontology


if __name__ == "__main__":
    # open spark session
    spark = SparkSession.builder.appName('translator').getOrCreate()

    # get the magma genes df
    df_magma_genes = process_magma(spark, magma_srcdir, ncbi_file)
    print("magma genes df: \n{}".format(df_magma_genes.show(10)))
    print("magma genes df of size: {}".format(df_magma_genes.count()))

    # join with the phenotype ontology
    df_magma_phenotypes = join_magma_phenotype_ontologies(spark, df_magma_genes, phenotype_file)
    print("magma phenotype df: \n{}".format(df_magma_phenotypes.show(10)))
    print("magma phenotype df of size: {}".format(df_magma_phenotypes.count()))

    # convert the disease/phenotype category to biolink notation
    df_magma_phenotypes = df_magma_phenotypes.withColumn("biolink_category",when(col("category").isin(['Disease']), category_disease).otherwise(category_phenotype))
    print("magma phenotype modified df: \n")
    df_magma_phenotypes.show(10)

    # filter the result for writing 
    df_magma_phenotypes = df_magma_phenotypes.select(
        df_magma_phenotypes.phenotype.alias('phenotype_code'),
        df_magma_phenotypes.tran_lookup_id.alias('phenotype_ontology_id'),
        df_magma_phenotypes.tran_lookup_name.alias('phenotype'),
        df_magma_phenotypes.dichotomous,
        df_magma_phenotypes.biolink_category,
        df_magma_phenotypes.group_name,
        df_magma_phenotypes.ncbi_id,
        df_magma_phenotypes.gene,
        df_magma_phenotypes.pValue.alias('p_value'),
    )
    df_magma_phenotypes.show(10)

    # write out the tsv file
    df_magma_phenotypes.coalesce(1) \
        .write \
        .mode('overwrite') \
        .option("delimiter", "\t") \
        .csv(outdir, header='true')

    # log
    print("saved data to directory: {}".format(outdir))


