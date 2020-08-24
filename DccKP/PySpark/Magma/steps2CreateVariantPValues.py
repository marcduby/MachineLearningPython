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
variant_srcdir = '/home/javaprog/Data/Broad/Magma/Common/'
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
        .select('varId', 'chromosome', 'n', 'pValue', 'position')

def load_rsids(input_srcdir):
    # load the variants
    return spark.read \
        .csv('%s/part-*' % (input_srcdir), sep='\t', header=True, schema=variant_schema) \
        .select('varId', 'dbSNP')# method to load the rdIds


# open spark session
spark = SparkSession.builder.appName('bioindex').getOrCreate()
print("got Spark session of type {}".format(type(spark)))

# load the variants pValues
df_load = load_pvalues(phenotype, pvalue_srcdir)

# print
print("the loaded variant pValue data frame has {} rows".format(df_load.count()))
df_load.show()
        
# load the variants pValues
df_variant_load = load_rsids(variant_srcdir)

# print
print("the loaded variant data frame has {} rows".format(df_variant_load.count()))
df_variant_load.show()


# # keep only the rows with non null dbSNP ids
# df_nonnull_load = df_load.filter(col("dbSNP").isNotNull())

# # print
# print("the non null RS id dataframe has {} rows".format(df_nonnull_load.count()))


# # %%
# df_nonnull_load.show()


# # %%
# # decompose first field and get chrom/pos
# split_col = split(df_nonnull_load['varId'], ':')

# # add the first two columns back in
# df_nonnull_load = df_nonnull_load.withColumn('chromosome', split_col.getItem(0))
# df_nonnull_load = df_nonnull_load.withColumn('position', split_col.getItem(1))


# # %%
# df_nonnull_load.show()


# # %%
# # build out data frame and save magma variant input file
# df_export = df_nonnull_load.select("dbSnp", 'chromosome', 'position')


# # %%
# df_export.count()


# # %%
# # replace the X/Y chromosome values with 23/24
# df_export = df_export.withColumn('chromosome', regexp_replace('chromosome', 'X', '23'))
# df_export = df_export.withColumn('chromosome', regexp_replace('chromosome', 'Y', '24'))


# # %%
# df_export.count()


# # %%
# df_export.printSchema()


# # %%
# # show the counts
# df_export.groupBy("chromosome").count().orderBy("chromosome").show(25, False)


# # %%
# df_export = df_export.filter(col("chromosome") != 'MT')


# # %%
# # show the counts
# df_export.groupBy("chromosome").count().orderBy("chromosome").show(25, False)


# # %%
# # write by chromosome
# for chrom in range(1, 3):
#     df_write = df_export.filter(col('chromosome') == chrom)
#     # write out the tab delimited file
#     print("chrom {} has size {}".format(chrom, df_write.count()))
#     df_write.write.mode('overwrite').option("delimiter", "\t").csv(out_dir + "/" + str(chrom))


# # %%
# # write out by chrom
# df_export.write.mode('overwrite').option("delimiter", "\t").partitionBy("chromosome").saveAsTable(out_dir)


# # %%
# # write out the tab delimited file
# df_export.write.mode('overwrite').option("delimiter", "\t").csv(out_dir)


# # %%

# # example

# #    by_phenotype.drop(['rank', 'top']) \
# #         .orderBy(['phenotype', 'pValue']) \
# #         .write \
# #         .mode('overwrite') \
# #         .json('%s/phenotype' % outdir)

