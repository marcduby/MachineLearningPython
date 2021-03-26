# imports
# from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, IntegerType
# from pyspark.sql.functions import col, struct, explode, when, lit, array_max, array, split, regexp_replace
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType


def main():
    """
    Arguments: phenotype
    """
    opts = argparse.ArgumentParser()
    opts.add_argument('phenotype')

    # parse command line
    args = opts.parse_args()

    # input and output directories
    dir_s3 = f's3://dig-analysis-data/out'
    dir_s3 = f'/home/javaprog/Data/Broad/dig-analysis-data/out'
    dir_s3 = f'/Users/mduby/Data/Broad/dig-analysis-data/out'
    dir_snp = f'{dir_s3}/varianteffect/snp'
    dir_common = f'{dir_s3}/metaanalysis/trans-ethnic/{args.phenotype}'
    dir_clump = f'{dir_s3}/metaanalysis/clumped/{args.phenotype}'
    dir_out = f'{dir_s3}/finemapping/{args.phenotype}'

    # start spark
    spark = SparkSession.builder.appName('cojo').getOrCreate()

    # load the snps
    df_snp = spark.read.csv(f'{dir_snp}/*.csv', sep='\t', header=True)
    print("got snps df of size {}".format(df_snp.count()))

    # load variants and phenotype associations
    # df_common = spark.read.json(f'{dir_common}/part-*')
    # df_common.show()

    # df_clump = spark.read.json(f'{dir_clump}/part-*')
    # df_clump.show()

    # first join on rsID
    # # drop variants with no dbSNP and join
    # snps = snps.filter(snps.dbSNP.isNotNull())

    # next join on metaanalysis to get pValue, stdErr, subjects
    # join to get the rsID for each
    # df = df.join(snps, on='varId')

    # columns are:
    # SNP 
    # A1 - the effect allele (alt)
    # A2 - the other allele (ref) 
    # freq - frequency of the effect allele 
    # b - effect size
    # se - standard error
    # p - p-value 
    # N - sample size
    # # keep only the columns magma needs in the correct order
    # df = df.select(
    #     df.dbSNP.alias('SNP'),
    #     df.pValue.alias('P'),
    #     df.n.cast(IntegerType()).alias('subjects'),
    # )

    # output results
    # df.write \
    #     .mode('overwrite') \
    #     .csv(dir_out, sep='\t', header='true')

    # done
    spark.stop()


if __name__ == '__main__':
    main()
