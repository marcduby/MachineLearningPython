# imports
import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType
from pyspark.sql.functions import lit, col, split, input_file_name, regexp_extract


# NCBI_SRC = 's3://dig-analysis-data/bin/magma/NCBI37.3.gene.loc'
# NCBI_SCHEMA = StructType() \
#     .add("geneNcbiId", IntegerType(), True) \
#     .add("chromosome", StringType(), True) \
#     .add("start", IntegerType(), True) \
#     .add("end", IntegerType(), True) \
#     .add("direction", StringType(), True) \
#     .add("gene", StringType(), True)

# constants
file_src = "/home/javaprog/Data/Broad/dig-analysis-data/out/magma/staging/pathways/*/associations.pathways.gsa.out"
dir_out = ""

def main():
    # """
    # Arguments: phenotype
    # """
    # opts = argparse.ArgumentParser()
    # opts.add_argument('phenotype')

    # # parse command line
    # args = opts.parse_args()

    # start spark session
    spark = SparkSession.builder.appName('magma').getOrCreate()

    # # EC2 development localhost directories
    # srcfile = f's3://dig-analysis-data/out/magma/staging/genes/{args.phenotype}/associations.genes.out'
    # outdir = f's3://dig-analysis-data/out/magma/gene-associations/{args.phenotype}'

    # NOTE: This file is whitespace-delimited, which Spark can't properly read.
    #       For this reason, we load it with a dummy delimiter and only get a
    #       single column back. We then split the columns and cast them to the
    #       correct type.

    df = spark.read.csv(file_src, sep='^', header=True, comment='#') \
    .withColumn('filename', input_file_name()) \
    .withColumn('phenotype', regexp_extract('filename', r'pathways/([^/]+)/associations', 1))
    # .filter(lambda line: len(line)>=4)
    # .withColumn('phenotype', regexp_extract('filename', r'pathways/([^/]+)/associations', 1))
    df.show()

    df = df.select(split(df[0], r'\s+'))
    df = df.select(
        df[0][7].alias('pathway_name').cast(StringType()),
        df[0][2].alias('number_genes').cast(IntegerType()),
        df[0][3].alias('beta').cast(DoubleType()),
        df[0][4].alias('beta_std').cast(DoubleType()),
        df[0][5].alias('standard_error').cast(DoubleType()),
        df[0][6].alias('pValue').cast(DoubleType()),
        col('phenotype')
    ) 
    df.show()

    # # join the NCBI data with the gene output
    # df = df.join(ncbi, on='geneNcbiId')
    # df = df.select(
    #     df.gene,
    #     lit(args.phenotype).alias('phenotype'),
    #     df.nParam,
    #     df.subjects,
    #     df.zStat,
    #     df.pValue,
    # )

    # # write the results
    # df.write.mode('overwrite').json(outdir)

    # done
    spark.stop()


if __name__ == '__main__':
    main()
