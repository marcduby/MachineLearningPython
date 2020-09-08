# imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, IntegerType
from pyspark.sql.functions import col, struct, explode, when, lit, array_max, array, split, regexp_replace

# EC2 development localhost directories
# ncbi_srcdir = '/home/javaprog/dig-analysis-data/bin/magma/'
# gene_pvalues_srcdir = '/home/javaprog/dig-analysis-data/out/magma/step4GenePValues/'
# out_dir = '/home/javaprog/Data/Broad/Magma/Out/Step3'

# development localhost directories
# variant_srcdir = '/Users/mduby/Data/Broad/Magma/Snp/'
# out_dir = '/Users/mduby/Data/Broad/Magma/Out/Step1'

# localhost development localhost directories
ncbi_srcdir = '/home/javaprog/Data/Broad/dig-analysis-data/bin/magma'
gene_pvalues_srcdir = '/home/javaprog/Data/Broad/dig-analysis-data/out/magma/step4GenePValues'
out_dir = '/home/javaprog/Data/Broad/dig-analysis-data/out/magma/results'

# ncbi schema
ncbi_schema = StructType([
    StructField("geneId", IntegerType(), True),
    StructField("chromosome", StringType(), True),
    StructField("start", IntegerType(), True),
    StructField("end", IntegerType(), True),
    StructField("direction", StringType(), True),
    StructField("gene", StringType(), True)
    ])

# this is the schema written out by the frequency analysis processor
gene_pvalue_schema = StructType(
    [
        # StructField('gene', IntegerType(), nullable=False),
        StructField('chromosome', StringType(), nullable=False),
        # StructField('start', IntegerType(), nullable=False),
        # StructField('stop', IntegerType(), nullable=False),
        # StructField('nsnps', IntegerType(), nullable=False),
        # StructField('nParam', IntegerType(), nullable=False),
        # StructField('subjects', IntegerType(), nullable=False),
        # StructField('zStat', DoubleType(), nullable=False),
        # StructField('pValue', DoubleType(), nullable=False),
    ]
)

# print
print("the ncbi input directory is: {}".format(ncbi_srcdir))
print("the output directory is: {}".format(out_dir))

# open spark session
spark = SparkSession.builder.appName('magma01').getOrCreate()
print("got Spark session of type {}".format(type(spark)))

# method to load the ncbi lookup file
def load_ncbi_lookup(input_dir):
    df = spark.read.csv('%s/NCBI37.3.gene.loc' % (input_dir), sep='\t', header=False, schema=ncbi_schema)

    # return
    return df

# method to load the gene pValues files
def load_gene_pvalues(input_dir, phenotype):
    # df = spark.read.csv('%s/%s/genePValues.txt' % (input_dir, phenotype), sep='\\s+', ignoreLeadingWhiteSpace=True, header=True)
    # df = DataFrame(load('%s/%s/genePValues.txt' % (input_dir, phenotype), spacedelim=true))
    # rdd = spark.textFile('%s/%s/genePValues.txt' % (input_dir, phenotype)).map(line=>line.split("\\s+"))
    # df = spark.read.option("header","true")\
    #     .option("delimiter"," ")\
    #     .csv('%s/%s/genePValues.txt' % (input_dir, phenotype)) 

    # load the dataframe as one line due to dynamic space delimiter
    df_load = spark.read.csv('%s/%s/genePValues.txt' % (input_dir, phenotype), schema=gene_pvalue_schema, header=True)

    # now split the columns
    split_col = split(df_load.chromosome, '\\s+',)
    df = df_load.withColumn("geneId", split_col.getItem(0))\
        .withColumn("nParam", split_col.getItem(5))\
        .withColumn("subjects", split_col.getItem(6))\
        .withColumn("zStat", split_col.getItem(7))\
        .withColumn("pValue", split_col.getItem(8))

    # return
    return df

# load the lookup file
df_ncbi = load_ncbi_lookup(ncbi_srcdir)
df_ncbi.show()

# load the gene pvalue file
# rdd_pValue = load_gene_pvalues(gene_pvalues_srcdir, 'BMI')
# print("loaded rdd")
# df_pValue = spark.createDataFrame(rdd).toDF(schema=gene_pvalue_schema)

# load the gene pvalue file
phenotype = 'BMI'
df_pvalue = load_gene_pvalues(gene_pvalues_srcdir, phenotype)
df_pvalue.show()

# join the two dataframes and add in rsIDs
df_export = df_pvalue.join(df_ncbi, on='geneId', how='inner')
df_export = df_export.select('geneId', 'nParam', 'subjects', 'zStat', 'pValue', 'gene').withColumn('phenotype', lit(phenotype))
print("the loaded variant joined data frame has {} rows".format(df_export.count()))
df_export.show()

# write out the one tab delimited file
# df_export.coalesce(1).write.mode('overwrite').option("delimiter", "\t").csv('%s/%s' % (out_dir, phenotype), header='true')
# print("wrote out the loaded variant data frame to directory {}".format(out_dir))
df_export \
        .orderBy(df_export.gene) \
        .write \
        .mode('overwrite') \
        .json('%s/%s' % (out_dir, phenotype))

# stop spark
# spark.stop()

