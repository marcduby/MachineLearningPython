# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, IntegerType
from pyspark.sql.functions import col, struct, explode, when, lit, array_max, array, split, regexp_replace


# variant_srcdir = 's3://dig-analysis-data/out/varianteffect/common/part-*'
# outdir = 's3:/dig-analysis-data/out/varianteffect/magma/'

# EC2 development localhost directories
# variant_srcdir = '/Users/mduby/Data/Broad/Magma/Common/part*'
# pvalue_srcdir = '/Users/mduby/Data/Broad/Magma/Phenotype/'
# out_dir = '/Users/mduby/Data/Broad/Magma/Out/Step2'

# development localhost directories
variant_srcdir = '/home/javaprog/Data/Broad/Magma/Snp/'
pvalue_srcdir = '/home/javaprog/Data/Broad/Magma/Phenotype/'
out_dir = '/home/javaprog/Data/Broad/Magma/Out/Step2'

# common variables
phenotype = 'BMI'

# this is the schema for the common variant file
variant_schema = StructType(
    [
        StructField('varId', StringType(), nullable=False),
        StructField('dbSNP', StringType(), nullable=False),
        StructField('consequence', StringType(), nullable=False),
        StructField('gene', StringType(), nullable=False),
        StructField('transcript', StringType(), nullable=False),
        StructField('impact', StringType(), nullable=False),
    ]
)

# print
print("the variant pValues input directory is: {}".format(pvalue_srcdir))
print("the output directory is: {}".format(out_dir))

# functions
# method to load the frequencies
def load_pvalues(pehnotype, input_srcdir):
    return spark.read \
        .json('%s/%s/part-*' % (input_srcdir, phenotype)) \
        .select('varId', 'n', 'pValue')

def load_rsids(input_srcdir):
    # load the variants
    return spark.read \
        .csv('%s/part-*' % (input_srcdir), sep='\t', header=True) \
        .select('varId', 'dbSNP')# method to load the rdIds


# open spark session
spark = SparkSession.builder.appName('bioindex').getOrCreate()
print("got Spark session of type {}".format(type(spark)))

# load the variants pValues
df_pvalue_load = load_pvalues(phenotype, pvalue_srcdir)

# print
print("the loaded variant pValue data frame has {} rows".format(df_pvalue_load.count()))
df_pvalue_load.show()
        
# load the variants pValues
df_variant_load = load_rsids(variant_srcdir)

# print
print("the loaded variant data frame has {} rows".format(df_variant_load.count()))
df_variant_load.show()

# join the two dataframes and add in rsIDs
df_export = df_pvalue_load.join(df_variant_load, on='varId', how='inner')
df_export = df_export.select('dbSNP', 'pValue', 'n').withColumnRenamed('n', 'subjects')
print("the loaded variant joined data frame has {} rows".format(df_export.count()))
df_export.show()

# write out the one tab delimited file
df_export.coalesce(1).write.mode('overwrite').option("delimiter", "\t").csv(out_dir)
print("wrote out the loaded variant data frame to directory {}".format(out_dir))

# stop spark
spark.stop()

