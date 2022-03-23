from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, input_file_name, regexp_extract, lit

def main():
    """
    Arguments: phenotype
    """
    # input and output directories
    # dir_s3 = f'/Users/mduby/Data/Broad/dig-analysis-data/out'
    dir_s3 = f'/home/javaprog/Data/Broad/dig-analysis-data/out'
    # dir_s3 = f's3://dig-analysis-data/out'
    dir_snp = f'{dir_s3}/varianteffect/snp'
    dir_largest_datasets = f'{dir_s3}/finemapping/largest-datasets'
    # dir_frequency = f'{dir_s3}/finemapping/variant-frequencies'
    dir_out = "{}/finemapping/variant-associations-largest-datasets/{}"

    # start spark
    spark = SparkSession.builder.appName('cojo').getOrCreate()

    # load the snps
    df_snp = spark.read.csv(f'{dir_snp}/*.csv', sep='\t', header=True)
    print("got snps df of size {}".format(df_snp.count()))
    df_snp.show(5)

    # remove variants with no dbSNP 
    df_snp = df_snp \
        .filter(df_snp.dbSNP.isNotNull()) \
        .select(
            df_snp.varId,
            df_snp.dbSNP,
        )
    print("got snps with non null dbSNP df of size {}".format(df_snp.count()))
    df_snp.show(5)

    # done
    spark.stop()


if __name__ == '__main__':
    main()
