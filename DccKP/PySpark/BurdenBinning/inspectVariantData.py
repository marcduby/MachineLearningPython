# imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, IntegerType
from pyspark.sql.functions import col, struct, explode, when, lit, array, udf

# load and output directory
# vep_srcdir = 's3://dig-analysis-data/out/varianteffect/effects/part-*'
# freq_srcdir = 's3://dig-analysis-data/out/frequencyanalysis/'
# outdir = 's3://dig-bio-index/burden/variantgene'

# development localhost directories
vep_srcdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/varianteffect/effects/part-*'
freq_srcdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/frequencyanalysis/'
outdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/burdenbinning/results'

# print
# print("the input directory is: {}".format(vep_srcdir))

# open spark session
spark = SparkSession.builder.appName('burdenbinning').getOrCreate()

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
gene_ensemble_id = "ensemblId"
burden_bin_id = "burdenBinId"
maf = 'maf'

# column variables for output
var_id_col = col(var_id)
gene_ensemble_id_col = col(gene_ensemble_id)
burden_bin_id_col = col(burden_bin_id)
maf_col = col(maf)

# column variables for filters
filter_lof_col = col("lof")
filter_impact_col = col("impact")
filter_polyphen2_hdiv_pred_col = col("polyphen2_hdiv_pred")
filter_polyphen2_hvar_pred_col = col("polyphen2_hvar_pred")
filter_sift_pred_col = col("sift_pred")
filter_lrt_pred_col = col("lrt_pred")
filter_mutationtaster_pred_col = col("mutationtaster_pred")

filter_metalr_pred_col = col("metalr_pred")
filter_provean_pred_col = col("provean_pred")
filter_fathmm_pred_col = col("fathmm_pred")
filter_fathmm_mkl_coding_pred_col = col("fathmm_mkl_coding_pred")
filter_eigen_pc_raw_rankscore_col = col("eigen_pc_raw_rankscore")
filter_dann_rankscore_col = col("dann_rankscore")
filter_vest3_rankscore_col = col("vest3_rankscore")
filter_cadd_raw_rankscore_col = col("cadd_raw_rankscore")
filter_metasvm_pred_col = col("metasvm_pred")

# variables for filters conditions
condition_lof_hc = filter_lof_col == 'HC'
condition_impact_moderate = (filter_impact_col == 'MODERATE') & (maf_col < 0.01)
condition_impact_high = (filter_impact_col == 'HIGH') & (filter_lof_col == 'HC') & (maf_col < 0.01)
# condition_impact_moderate = filter_impact_col == 'MODERATE'
# condition_impact_high = filter_impact_col == 'HIGH'

# level 2 condition for bin 7
condition_level2_bin7 = (filter_polyphen2_hdiv_pred_col != 'D') & \
        (filter_polyphen2_hvar_pred_col != 'D') & \
        (filter_sift_pred_col != 'deleterious') &  \
        (filter_lrt_pred_col != 'D') & \
        (~filter_mutationtaster_pred_col.isin(['A', 'D']))

# level 2 exclusion condition for bin 6
condition_level2_inclusion_bin6 = (filter_polyphen2_hdiv_pred_col == 'D') | \
        (filter_polyphen2_hvar_pred_col == 'D') | \
        (filter_sift_pred_col == 'deleterious') | \
        (filter_lrt_pred_col == 'D') | \
        (filter_mutationtaster_pred_col.isin(['A', 'D']))

# level 2 exclusion condition for bin 5
condition_level2_inclusion_bin5 = (filter_polyphen2_hdiv_pred_col == 'D') & \
        (filter_polyphen2_hvar_pred_col == 'D') & \
        (filter_sift_pred_col == 'deleterious') & \
        (filter_lrt_pred_col == 'D') & \
        (filter_mutationtaster_pred_col.isin(['A', 'D']))

# level 2 exclusion condition for bin 3
condition_level2_inclusion_bin3 = condition_level2_inclusion_bin5 & \
        (filter_metalr_pred_col == 'D') & \
        (filter_metasvm_pred_col == 'D') &  \
        (filter_provean_pred_col == 'D') & \
        (filter_fathmm_mkl_coding_pred_col == 'D') & \
        (filter_fathmm_pred_col == 'D')

# level 2 exclusion condition for bin 2
condition_level2_inclusion_bin2 = condition_level2_inclusion_bin3 & \
        (filter_eigen_pc_raw_rankscore_col > 0.9) & \
        (filter_dann_rankscore_col > 0.9) & \
        (filter_cadd_raw_rankscore_col > 0.9) & \
        (filter_vest3_rankscore_col > 0.9) 

# schemas for csv files
# this is the schema written out by the frequency analysis processor
frequency_schema = StructType(
    [
        StructField('varId', StringType(), nullable=False),
        StructField('chromosome', StringType(), nullable=False),
        StructField('position', IntegerType(), nullable=False),
        StructField('reference', StringType(), nullable=False),
        StructField('alt', StringType(), nullable=False),
        StructField('eaf', DoubleType(), nullable=False),
        StructField('maf', DoubleType(), nullable=False),
        StructField('ancestry', StringType(), nullable=False),
    ]
)

# functions
# method to load the frequencies
# method to load the frequencies
def load_freq(ancestry_name, input_srcdir):
    return spark.read \
        .json('%s/%s/part-*' % (input_srcdir, ancestry_name)) \
        .select(var_id_col, maf_col.alias(ancestry_name))

# method to get the max of an array
def max_array(array_var):
    max = 0.0                        # maf will never be less than 0
    for element in array_var:
        if (element is not None):
            if (element > max):
                max = element
    return max

# custom function used for sorting chromosomes properly
max_array_udf = udf(max_array, DoubleType())


# # load and do the maf calculations
# # frequency outputs by ancestry
# # ancestries = ['AA', 'AF', 'EA', 'EU', 'HS', 'SA']
# ancestries = ['AA', 'EA', 'EU', 'HS', 'SA']
# dataframe_freq = None

# # load frequencies by variant ID
# for ancestry in ancestries:
#     df = load_freq(ancestry, freq_srcdir)

#     # final, joined frequencies
#     dataframe_freq = df if dataframe_freq is None else dataframe_freq.join(df, var_id, how='outer')

# # pull all the frequencies together into a single array
# dataframe_freq = dataframe_freq.select(dataframe_freq.varId, array(*ancestries).alias('frequency'))
# #
# # # get the max for all frequencies
# dataframe_freq = dataframe_freq.withColumn('maf', max_array_udf('frequency')).select(dataframe_freq.varId, 'maf')

# # print
# print("the loaded frequency data frame has {} rows".format(dataframe_freq.count()))
# dataframe_freq.show()

# load the transcript json data
vep = spark.read.json(vep_srcdir)
vep.show()

# create new data frame with only var id
transcript_consequences = vep.select(vep.id, vep.transcript_consequences)     .withColumn('cqs', explode(col('transcript_consequences')))     .select(
        col('id').alias('varId'),
        col('cqs.gene_id').alias(gene_ensemble_id),
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
transcript_consequences.show()

# pull only 2 good variants and display
df_var = transcript_consequences.filter(col('varId').isin(["10:100160008:C:T", "10:11791527:T:TA"]))
df_var.show()

