# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, IntegerType
from pyspark.sql.functions import col, struct, explode, when, lit


# %%
# load and output directory
# vep_srcdir = 's3://dig-analysis-data/out/varianteffect/effects/part-*'
# outdir = 's3://dig-bio-index/burden/vepbinning'

# test directories
vep_srcdir = '/Users/mduby/Data/Broad/Aggregator/BurdenBinning/20200330/test*'
outdir = '/Users/mduby/Data/Broad/Aggregator/BurdenBinning/20200330/Out'

# print
print("the input directory is: {}".format(vep_srcdir))


# %%
# constants for filters
# there are 3 levels of filters (lof, impact + maf, and combined predictions)
# the 7 bins will combine variantions of these three OR conditions

# general filter
filter_pick = "pick"

# level 1 filter
filter_lof = "lof"

# level 2 filters
filter_polyphen2_hdiv_pred = "polyphen2_hdiv_pred"
filter_polyphen2_hvar_pred = "polyphen2_hvar_pred"
filter_sift_red = "sift_pred"
filter_mutationtaster_pred = "mutationtaster_pred"
filter_lrt_pred = "lrt_pred"
filter_metalr_pred = "metalr_pred"
filter_provean_pred = "provean_pred"
filter_fathmm_pred = "fathmm_pred"
filter_fathmm_mkl_coding_pred = "fathmm-mkl_coding_pred"
filter_eigen_pc_raw_rankscore = "eigen-pc-raw_rankscore"
filter_dann_rankscore = "dann_rankscore"
filter_vest3_rankscore = "vest3_rankscore"
filter_cadd_raw_rankscore = "cadd_raw_rankscore"
filter_metasvm_pred = "metasvm_pred"

# aliases w/o -
filter_fathmm_mkl_coding_pred_alias = "fathmm_mkl_coding_pred"
filter_eigen_pc_raw_rankscore_alias = "eigen_pc_raw_rankscore"

# level 3 filter
filter_impact = "impact"

# column constants
var_id = "varId"
gene_ensemble_id = "geneEnsembleId"
burden_bin_id = "burden_bin_id"


# %%
# variant list schema
all_schema = StructType(
    [
        StructField('gene_name', StringType(), nullable=False),
        StructField('gene_ensemble_id', StringType(), nullable=False),
        StructField('chromosome', StringType(), nullable=False),
        StructField('position', IntegerType(), nullable=False),
        StructField('burdenBinId', IntegerType(), nullable=False),
        StructField('varId', StringType(), nullable=False),
    ]
)


# %%
# open spark session
spark = SparkSession.builder.appName('bioindex').getOrCreate()


# %%
# load the json data
vep = spark.read.json(vep_srcdir)

# print
print("the loaded vep data count is: {}".format(vep.count()))
# format(vep.show())


# %%
# create new data frame with only var id
transcript_consequences = vep.select(vep.id, vep.transcript_consequences)     .withColumn('cqs', explode(col('transcript_consequences')))     .select(
        col('id').alias('varId'),
        col('cqs.gene_id').alias('geneEnsembleId'),
        col('cqs.' + filter_lof).alias(filter_lof),
        col('cqs.' + filter_impact).alias(filter_impact),

        col('cqs.' + filter_polyphen2_hdiv_pred).alias(filter_polyphen2_hdiv_pred),
        col('cqs.' + filter_polyphen2_hvar_pred).alias(filter_polyphen2_hvar_pred),
        col('cqs.' + filter_sift_red).alias(filter_sift_red),
        col('cqs.' + filter_mutationtaster_pred).alias(filter_mutationtaster_pred),
        col('cqs.' + filter_lrt_pred).alias(filter_lrt_pred),
        col('cqs.' + filter_metalr_pred).alias(filter_metalr_pred),

        col('cqs.' + filter_provean_pred).alias(filter_provean_pred),
        col('cqs.' + filter_fathmm_pred).alias(filter_fathmm_pred),
        col('cqs.' + filter_fathmm_mkl_coding_pred).alias(filter_fathmm_mkl_coding_pred_alias),
        col('cqs.' + filter_eigen_pc_raw_rankscore).alias(filter_eigen_pc_raw_rankscore_alias),
        col('cqs.' + filter_dann_rankscore).alias(filter_dann_rankscore),
        col('cqs.' + filter_vest3_rankscore).alias(filter_vest3_rankscore),
        col('cqs.' + filter_cadd_raw_rankscore).alias(filter_cadd_raw_rankscore),
        col('cqs.' + filter_metasvm_pred).alias(filter_metasvm_pred)
    )


# print
print("the filtered transcript consequence data count is: {}".format(transcript_consequences.count()))
# transcript_consequences.show()


# %%
# get the lof level 1 data frame
dataframe_lof = transcript_consequences.filter(transcript_consequences.lof == 'HC')

# print
print("the lof data frame count is: {}".format(dataframe_lof.count()))
# dataframe_lof.show()


# %%
# get the level 3 dataframe
dataframe_impact_moderate = transcript_consequences.filter(transcript_consequences.impact == 'MODERATE')
dataframe_impact_high = transcript_consequences.filter(transcript_consequences.impact == 'HIGH')

# print
print("the moderate impact dataframe is {}".format(dataframe_impact_moderate.count()))
print("the high impact dataframe is {}".format(dataframe_impact_high.count()))
# dataframe_impact.show()


# %%
# get the initial level 2 dataframe
dataframe_level2 = transcript_consequences.filter(transcript_consequences.polyphen2_hdiv_pred != 'D')     .filter(transcript_consequences.polyphen2_hvar_pred != 'D')     .filter(transcript_consequences.sift_pred != 'deleterious')     .filter(transcript_consequences.lrt_pred != 'D')     .filter(~transcript_consequences.mutationtaster_pred.isin(['A', 'D']))

print("level 2 data frame count: {}".format(dataframe_level2.count()))
# dataframe_level2.show()


# %%
# BIN 1 of 7
# create the final_1 df, just lof = HC
final_bin1_data_frame = dataframe_lof.withColumn(burden_bin_id, lit('bin1_7'))

# print
print("the final bin 1 dataframe is: {}".format(final_bin1_data_frame.count()))
# final_bin1_data_frame.show()


# %%
# BIN 7 of 7
# get the initial level 2 dataframe
dataframe_level2 = transcript_consequences.filter(transcript_consequences.polyphen2_hdiv_pred != 'D')     .filter(transcript_consequences.polyphen2_hvar_pred != 'D')     .filter(transcript_consequences.sift_pred != 'deleterious')     .filter(transcript_consequences.lrt_pred != 'D')     .filter(~transcript_consequences.mutationtaster_pred.isin(['A', 'D']))

print("level 2 data frame count: {}".format(dataframe_level2.count()))
print("moderate impact data frame count: {}".format(dataframe_impact_moderate.count()))
print("lof data frame count: {}".format(dataframe_lof.count()))
# dataframe_level2.show()

# create the final_7 df, lof = HC, impact moderate, add in level 2 filters
final_bin7_data_frame = dataframe_lof.union(dataframe_impact_moderate).union(dataframe_level2).distinct()
final_bin7_data_frame = final_bin7_data_frame.withColumn(burden_bin_id, lit('bin7_7'))

# print
print("the final bin 7 dataframe is: {}".format(final_bin7_data_frame.count()))
# final_bin7_data_frame.show()


# %%
# BIN 6 of 7
# get the exclusion level 2 data frame
dataframe_level2_exclusion = transcript_consequences     .filter(transcript_consequences.polyphen2_hdiv_pred == 'D')     .filter(transcript_consequences.polyphen2_hvar_pred == 'D')     .filter(transcript_consequences.sift_pred == 'deleterious')     .filter(transcript_consequences.lrt_pred == 'D')     .filter(transcript_consequences.mutationtaster_pred.isin(['A', 'D']))

dataframe_level2_inclusion = transcript_consequences     .filter(transcript_consequences.polyphen2_hdiv_pred == 'D')     .union(transcript_consequences.filter(transcript_consequences.polyphen2_hvar_pred == 'D'))     .union(transcript_consequences.filter(transcript_consequences.sift_pred == 'deleterious'))     .union(transcript_consequences.filter(transcript_consequences.lrt_pred == 'D'))     .union(transcript_consequences.filter(transcript_consequences.mutationtaster_pred.isin(['A', 'D'])))     .distinct()

print("level 2 exclusion data frame count: {}".format(dataframe_level2_exclusion.count()))
print("level 2 inclusion data frame count: {}".format(dataframe_level2_inclusion.count()))
print("moderate impact data frame count: {}".format(dataframe_impact_moderate.count()))
print("lof data frame count: {}".format(dataframe_lof.count()))
# dataframe_level2.show()

# create the final_6 df, lof = HC, impact moderate, add in level 2 filters
final_bin6_data_frame = dataframe_level2_inclusion.subtract(dataframe_level2_exclusion)
final_bin6_data_frame = final_bin6_data_frame.union(dataframe_lof).union(dataframe_impact_moderate).union(dataframe_level2_inclusion).distinct()
final_bin6_data_frame = final_bin6_data_frame.withColumn(burden_bin_id, lit('bin6_7'))

# print
print("the final bin 6 dataframe is: {}".format(final_bin6_data_frame.count()))
# final_bin7_data_frame.show()


# %%
# BIN 5 of 7
# already have the inclusion level 2 data frame (exclusion from the previous bin 6 of 7)

print("level 2 inclusion data frame count: {}".format(dataframe_level2_exclusion.count()))
print("high impact data frame count: {}".format(dataframe_impact_high.count()))
print("lof data frame count: {}".format(dataframe_lof.count()))
# dataframe_level2.show()

# create the final_5 df, lof = HC, impact moderate, add in level 2 filters
final_bin5_data_frame = dataframe_lof.union(dataframe_level2_exclusion).union(dataframe_impact_high).distinct()
final_bin5_data_frame = final_bin5_data_frame.withColumn(burden_bin_id, lit('bin5_7'))

# print
print("the final bin 5 dataframe is: {}".format(final_bin5_data_frame.count()))
# final_bin7_data_frame.show()


# %%
# BIN 4 of 7
# already have the inclusion level 2 data frame (exclusion from the previous bin 6 of 7)

print("level 2 inclusion data frame count: {}".format(dataframe_level2_exclusion.count()))
print("lof data frame count: {}".format(dataframe_lof.count()))
# dataframe_level2.show()

# create the final_4 df, lof = HC, impact moderate, add in level 2 filters
final_bin4_data_frame = dataframe_lof.union(dataframe_level2_exclusion).distinct()
final_bin4_data_frame = final_bin4_data_frame.withColumn(burden_bin_id, lit('bin4_7'))

# print
print("the final bin 4 dataframe is: {}".format(final_bin4_data_frame.count()))
# final_bin7_data_frame.show()


# %%
# BIN 3 of 7
# bin consists of bin4 level 2 filter with some added on filters
dataframe_bin3_level2_inclusion = dataframe_level2_exclusion     .filter(dataframe_level2_exclusion.metalr_pred == 'D')     .filter(dataframe_level2_exclusion.metasvm_pred == 'D')     .filter(dataframe_level2_exclusion.provean_pred == 'D')     .filter(dataframe_level2_exclusion.fathmm_mkl_coding_pred == 'D')     .filter(dataframe_level2_exclusion.fathmm_pred == 'D')

print("bin 3 level 2 inclusion data frame count: {}".format(dataframe_bin3_level2_inclusion.count()))
print("lof data frame count: {}".format(dataframe_lof.count()))
# dataframe_level2.show()

# create the final_3 df, lof = HC, add in level 2 filters
final_bin3_data_frame = dataframe_lof.union(dataframe_bin3_level2_inclusion).distinct()
final_bin3_data_frame = final_bin3_data_frame.withColumn(burden_bin_id, lit('bin3_7'))

# print
print("the final bin 3 dataframe is: {}".format(final_bin3_data_frame.count()))
# final_bin7_data_frame.show()


# %%
dataframe_bin3_level2_inclusion.show()


# %%
# BIN 2 of 7
# bin consists of bin3 level 2 filter with some more added on filters
dataframe_bin2_level2_inclusion = dataframe_bin3_level2_inclusion     .filter(dataframe_level2_exclusion.eigen_pc_raw_rankscore > 0.9)     .filter(dataframe_level2_exclusion.dann_rankscore > 0.9)     .filter(dataframe_level2_exclusion.cadd_raw_rankscore > 0.9)     .filter(dataframe_level2_exclusion.vest3_rankscore > 0.9)

print("bin 2 level 2 inclusion data frame count: {}".format(dataframe_bin2_level2_inclusion.count()))
print("lof data frame count: {}".format(dataframe_lof.count()))
# dataframe_level2.show()

# create the final_2 df, lof = HC, add in level 2 filters
final_bin2_data_frame = dataframe_lof.union(dataframe_bin2_level2_inclusion).distinct()
final_bin2_data_frame = final_bin2_data_frame.withColumn(burden_bin_id, lit('bin2_7'))

# print
print("the final bin 3 dataframe is: {}".format(final_bin3_data_frame.count()))
# final_bin7_data_frame.show()


# %%
# combine all the bins into one dataframe
output_data_frame = final_bin1_data_frame     .union(final_bin2_data_frame)     .union(final_bin3_data_frame)     .union(final_bin4_data_frame)     .union(final_bin5_data_frame)     .union(final_bin6_data_frame)     .union(final_bin7_data_frame).distinct().orderBy(var_id, gene_ensemble_id, burden_bin_id)
    # .distinct() \
    # .orderBy(var_id, gene_ensemble_id, burden_bin_id)

# print
print("the final agregated bin dataframe is: {}".format(output_data_frame.count()))


# %%
# save out the output data frame to file
output_data_frame     .write     .mode('overwrite')     .json('%s/burden' % outdir)

# print
print("Printed out {} records to bioindex".format(output_data_frame.count()))


# %%
# done
spark.stop()


# %%
# filter_polyphen2_hdiv_pred = "polyphen2_hdiv_pred"
# filter_polyphen2_hvar_pred = "polyphen2_hvar_pred"
# filter_sift_red = "sift_pred"
# filter_mutationtaster_pred = "mutationtaster_pred"
# filter_lrt_pred = "lrt_pred"
# filter_metalr_pred = "metalr_pred"
# filter_provean_pred = "provean_pred"
# filter_fathmm_pred = "fathmm_pred"
# filter_fathmm_mkl_coding_pred = "fathmm_mkl_coding_pred"
# filter_eigen_pc_raw_rankscore = "eigen_pc_raw_rankscore"
# filter_dann_rankscore = "dann_rankscore"
# filter_vest3_rankscore = "vest3_rankscore"
# filter_cadd_raw_rankscore = "cadd_raw_rankscore"
# filter_metasvm_pred = "metasvm_pred"

# aliases w/o -
# filter_fathmm_mkl_coding_pred_alias = "fathmm_mkl_coding_pred"
# filter_eigen_pc_raw_rankscore_alias = "eigen_pc-raw_rankscore"

