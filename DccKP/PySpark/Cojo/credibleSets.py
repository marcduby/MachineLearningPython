from pyspark.sql import SparkSession, Row
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import col, input_file_name, regexp_extract, concat
from pyspark.sql.functions import lit
import argparse
import functools


def unionAll(dfs):
    return functools.reduce(lambda df1,df2: df1.union(df2.select(df1.columns)), dfs)

def main():
    # """
    # Arguments: phenotype
    # """
    # opts = argparse.ArgumentParser()
    # opts.add_argument('phenotype')

    # # parse command line
    # args = opts.parse_args()
    # phenotype = args.phenotype

    # input and output directories
    dir_s3 = f's3://dig-analysis-data/out'
    dir_s3 = f'/Users/mduby/Data/Broad/dig-analysis-data/out'
    dir_s3 = f'/home/javaprog/Data/Broad/dig-analysis-data/out'
    dir_results = f'{dir_s3}/finemapping/cojo-results'
    dir_out = f'{dir_s3}/finemapping/credible-set-results'

    # start spark
    spark = SparkSession.builder.appName('cojo').getOrCreate()

    # load the lead snps
    df_lead_snp = spark.read.csv(f'{dir_results}/*/*/*.jma.cojo', sep='\t', header=True) \
        .withColumn('filename', input_file_name()) \
        .withColumn('ancestry', regexp_extract('filename', r'/ancestry=([^/]+)/', 1)) \
        .withColumn('pheno', regexp_extract('filename', r'/([^/]+)/ancestry=', 1))
    print("got lead snp df of size {}".format(df_lead_snp.count()))
    df_lead_snp.groupBy('ancestry', 'pheno').count().show(70)

    # add extra conditional columns as null
    df_lead_snp = df_lead_snp \
        .withColumn('bC', lit(None).cast(StringType())) \
        .withColumn('bC_se', lit(None).cast(StringType())) \
        .withColumn('pC', lit(None).cast(StringType())) \
        .withColumn('signal_chromosome', df_lead_snp.Chr) \
        .withColumn('signal_position', df_lead_snp.bp) \
        .withColumn('credible_set', df_lead_snp.SNP)
    print("have column types \n{}".format(df_lead_snp.dtypes))
    print(df_lead_snp.show(20))


    # df_lead_snp = df_lead_snp.filter(df_lead_snp.bp < 2885636).filter(df_lead_snp.bp > 2808400).filter(df_lead_snp.Chr.isin(11)).filter(df_lead_snp.pheno.isin('T2D'))
    # df_lead_snp.groupBy('ancestry', 'pheno').count().show(70)

    
    df_lead_snp = df_lead_snp.select(
        df_lead_snp.SNP.alias('dbSNP'),
        df_lead_snp.Chr.alias('chromosome'),
        df_lead_snp.bp,
        df_lead_snp.bp.alias('position').cast(IntegerType()),
        df_lead_snp.refA.alias('ref'),
        df_lead_snp.freq.alias('maf'),
        df_lead_snp.ancestry,
        df_lead_snp.pheno,
        df_lead_snp.p.alias('pValue'),
    ).sort(df_lead_snp.bp)
    print("got lead snp df of size {}".format(df_lead_snp.count()))
    print(df_lead_snp.printSchema())

    # add column of lead snp
    # print(df_lead_snp.show(20))


    # sort by pValue, build distinct snp array and dataframe 
    variants = df_lead_snp.orderBy(['pValue']).collect()   # actually run plan
    topVariants = []
    bottomVariants = []
    count = 0
    while variants:
        best = variants[0]
        temp = best.asDict()
        temp['lead_snp'] = best['chromosome'] + "_" + best['bp'] + "_" + best['pheno']
        topVariants.append(Row(**temp))

        # remove all variants around best and put them into the other array
        tempArray = [v for v in variants if (abs(v['position'] - best['position']) <= 10000000) and (v['position'] != best['position']) and (v['chromosome'] == best['chromosome']) and (v['pheno'] == best['pheno'])]
        for row in tempArray:
            temp = row.asDict()
            temp['lead_snp'] = best['chromosome'] + "_" + best['bp'] + "_"+ best['pheno']
            newRow = Row(**temp)
            bottomVariants.append(newRow)

        # bottomVariants += tempArray
        # if count == 0:
        #     print("bottom {}".format(bottomVariants))
        count += count
        variants = [v for v in variants if (abs(v['position'] - best['position']) > 10000000) or (v['chromosome'] != best['chromosome']) or (v['pheno'] != best['pheno'])]

    # make a new dataframe with the resulting top variants
    df_new_lead_snp = spark.createDataFrame(topVariants)
    df_not_lead_snp = spark.createDataFrame(bottomVariants)
    print("got lead snp df of size {} and non lead snp of size {}".format(df_new_lead_snp.count(), df_not_lead_snp.count()))
    print(df_new_lead_snp.show(20))


    # join with the conditioned snps

    # # load the conditioned snps
    # df_conditioned_snp = spark.read.csv(f'{dir_results}/*/*/*.cma.cojo', sep='\t', header=True) \
    #     .withColumn('filename', input_file_name()) \
    #     .withColumn('ancestry', regexp_extract('filename', r'/ancestry=([^/]+)/', 1)) \
    #     .withColumn('pheno', regexp_extract('filename', r'/([^/]+)/ancestry=', 1))
    # print("got conditioned snp df of size {}".format(df_conditioned_snp.count()))
    # df_conditioned_snp.groupBy('ancestry').count().show(20)

    # # add extra conditional columns as null
    # df_conditioned_snp = df_conditioned_snp \
    #     .withColumn('bJ', lit(None).cast(StringType())) \
    #     .withColumn('bJ_se', lit(None).cast(StringType())) \
    #     .withColumn('pJ', lit(None).cast(StringType())) \
    #     .withColumn('LD_r', lit(None).cast(StringType()))

    # only keep conditioned SNPs that are within 5MB of a lean snp signal


#     print("have column types \n{}".format(df_conditioned_snp.dtypes))

#     # combine the two dataframes
#     df_all_snp = unionAll([df_lead_snp, df_conditioned_snp])
#     print("got all snp df of size {}".format(df_all_snp.count()))
#     df_all_snp.show(40)
#     df_all_snp.groupBy(['pheno', 'ancestry']).count().show(70)

#     # rename the columns
#     df_all_snp = df_all_snp.select(
#         df_all_snp.SNP.alias('dbSNP'),
#         df_all_snp.Chr.alias('chromosome'),
#         df_all_snp.bp.alias('position'),
#         df_all_snp.refA.alias('alt'),
#         df_all_snp.freq.alias('maf'),
#         df_all_snp.n,
#         df_all_snp.b.alias('beta'),
#         df_all_snp.se.alias('stdErr'),
#         df_all_snp.p.alias('pValue'),
#         df_all_snp.freq_geno.alias('mafGenotype'),
#         df_all_snp.bJ.alias('betaJoint'),
#         df_all_snp.bJ_se.alias('stdErrJoint'),
#         df_all_snp.pJ.alias('pValueJoint'),
#         df_all_snp.bC.alias('betaConditioned'),
#         df_all_snp.bC_se.alias('stdErrConditioned'),
#         df_all_snp.pC.alias('pValueConditioned'),
#         df_all_snp.pheno.alias('phenotype'),
#         df_all_snp.ancestry
#      )
#     df_all_snp.show(4)

# # cojo headers
# # +---+-----------+---------+----+------+---------+---------+-----------+-------+---------+---------+---------+------------+-----------+--------------------+--------+---------+---------+-----------+
# # |Chr|        SNP|       bp|refA|  freq|        b|       se|          p|      n|freq_geno|       bJ|    bJ_se|          pJ|       LD_r|            filename|ancestry|       bC|    bC_se|         pC|
# # +---+-----------+---------+----+------+---------+---------+-----------+-------+---------+---------+---------+------------+-----------+--------------------+--------+---------+---------+-----------+

#     # write out the file
#     df_all_snp \
#         .write.mode('overwrite') \
#         .json(dir_out)
#     print("wrote out data to directory {}".format(dir_out))


    # done
    spark.stop()


if __name__ == "__main__":
    main()