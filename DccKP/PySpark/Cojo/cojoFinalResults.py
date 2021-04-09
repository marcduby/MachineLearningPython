from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import col, input_file_name, regexp_extract
from pyspark.sql.functions import lit
import argparse
import functools


def unionAll(dfs):
    return functools.reduce(lambda df1,df2: df1.union(df2.select(df1.columns)), dfs)

def main():
    """
    Arguments: phenotype
    """
    opts = argparse.ArgumentParser()
    opts.add_argument('phenotype')

    # parse command line
    args = opts.parse_args()
    phenotype = args.phenotype

    # input and output directories
    dir_s3 = f's3://dig-analysis-data/out'
    dir_s3 = f'/Users/mduby/Data/Broad/dig-analysis-data/out'
    dir_s3 = f'/home/javaprog/Data/Broad/dig-analysis-data/out'
    dir_results = f'{dir_s3}/finemapping/cojo-results'
    dir_out = f'{dir_s3}/finemapping/variant-results'

    # start spark
    spark = SparkSession.builder.appName('cojo').getOrCreate()

    # load the lead snps
    df_lead_snp = spark.read.csv(f'{dir_results}/{phenotype}/*/*.jma.cojo', sep='\t', header=True) \
        .withColumn('filename', input_file_name()) \
        .withColumn('ancestry', regexp_extract('filename', r'/ancestry=([^/]+)/', 1))
    print("got lead snp df of size {}".format(df_lead_snp.count()))
    df_lead_snp.groupBy('ancestry').count().show(70)

    # add extra conditional columns as null
    df_lead_snp = df_lead_snp \
        .withColumn('bC', lit(None).cast(StringType())) \
        .withColumn('bC_se', lit(None).cast(StringType())) \
        .withColumn('pC', lit(None).cast(StringType()))
    print("have column types \n{}".format(df_lead_snp.dtypes))

    # load the conditioned snps
    df_conditioned_snp = spark.read.csv(f'{dir_results}/{phenotype}/*/*.cma.cojo', sep='\t', header=True) \
        .withColumn('filename', input_file_name()) \
        .withColumn('ancestry', regexp_extract('filename', r'/ancestry=([^/]+)/', 1))
    print("got conditioned snp df of size {}".format(df_conditioned_snp.count()))
    df_conditioned_snp.groupBy('ancestry').count().show(20)

    # add extra conditional columns as null
    df_conditioned_snp = df_conditioned_snp \
        .withColumn('bJ', lit(None).cast(StringType())) \
        .withColumn('bJ_se', lit(None).cast(StringType())) \
        .withColumn('pJ', lit(None).cast(StringType())) \
        .withColumn('LD_r', lit(None).cast(StringType()))

    print("have column types \n{}".format(df_conditioned_snp.dtypes))

    # combine the two dataframes
    df_all_snp = unionAll([df_lead_snp, df_conditioned_snp])
    print("got all snp df of size {}".format(df_all_snp.count()))
    df_all_snp.show(40)
    df_all_snp.groupBy('ancestry').count().show(70)

    # rename the columns
    # df = df.select(
    #     df.dbSNP.alias('SNP'),
    #     df.pValue.alias('P'),
    #     df.n.cast(IntegerType()).alias('subjects'),
    # )

# +---+-----------+---------+----+------+---------+---------+-----------+-------+---------+---------+---------+------------+-----------+--------------------+--------+---------+---------+-----------+
# |Chr|        SNP|       bp|refA|  freq|        b|       se|          p|      n|freq_geno|       bJ|    bJ_se|          pJ|       LD_r|            filename|ancestry|       bC|    bC_se|         pC|
# +---+-----------+---------+----+------+---------+---------+-----------+-------+---------+---------+---------+------------+-----------+--------------------+--------+---------+---------+-----------+


    # done
    spark.stop()


if __name__ == "__main__":
    main()