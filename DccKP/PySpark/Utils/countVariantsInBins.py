# imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, IntegerType
from pyspark.sql.functions import col, struct, explode, when, lit, array, udf

# load and output directory
# vep_srcdir = 's3://dig-analysis-data/out/varianteffect/effects/part-*'
# freq_srcdir = 's3://dig-analysis-data/out/frequencyanalysis/'
# outdir = 's3://dig-analysis-data/out/burdenbinning/results'

# development localhost directories
vep_srcdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/varianteffect/effects/part-*'
common_srcdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/varianteffect/common/part-*'
outdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/burdenbinning/results'

# print
# print("the input directory is: {}".format(vep_srcdir))

# open spark session
spark = SparkSession.builder.appName('variantcount').getOrCreate()

# load the transcript json data
vep = spark.read.json(vep_srcdir)

# create new data frame with only var id
vep_var_id = vep.select(vep.id, vep.transcript_consequences) \
    .withColumn('cqs', explode(col('transcript_consequences'))) \
    .select(
        col('id').alias('varId')
    )

# print
vep_var_id.show()
print("in directory {}".format(vep_srcdir))
print("the vep variant total count is: {}".format(vep_var_id.count()))
print("the vep variant distinct count is: {}\n\n".format(vep_var_id.distinct().count()))

# load the common variant json data
common = spark.read.json(common_srcdir)

# create new data frame with only var id
common_var_id = common.select(col('varId'))

# print
common_var_id.show()
print("in directory {}".format(common_srcdir))
print("the common variant total count is: {}".format(common_var_id.count()))
print("the common variant distinct count is: {}".format(common_var_id.distinct().count()))

# done
# spark.stop()

