
# STEP 03 - USE SPARK TO JOIN THE PHENOTYPE CURIES WITH THE ASSOCIATED ROWS IN THE 4GB PVALUE FILE

# imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, IntegerType
from pyspark.sql.functions import col, struct, explode, when, lit, array, udf

# constants
file_genebass = "/home/javaprog/Data/Broad/Translator/Genebass/genebassPValue.tsv"
# file_genebass = "/home/javaprog/Data/Broad/Translator/Genebass/test_genebassPValue.tsv"
# file_phenotypes = "/home/javaprog/Data/Broad/Translator/Genebass/ukBiobankCuries20210830.tsv"
file_phenotypes = "/home/javaprog/Data/Broad/Translator/Genebass/ukBiobankCuries20210902.tsv"
file_output = "/home/javaprog/Data/Broad/Translator/Genebass/Filtered/ukBiobankFilteredResults.tsv"
dir_filtered = "/home/javaprog/Data/Broad/Translator/Genebass/Filtered"

if __name__ == "__main__":
    # open spark session
    spark = SparkSession.builder.appName('genebass') \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.memory.fraction", 0.8) \
        .config("spark.executor.memory", "14g") \
        .config("spark.driver.memory", "12g")\
        .getOrCreate()

    # read the file
    df_phenotypes = spark.read.csv(file_phenotypes, sep='\t', header=True) \
        .select('phenocode', 'coding', 'description', 'coding_description', 'curie_id')
    df_phenotypes.toPandas().info()
    # df_phenotypes.show(2)

    # get distinct curies
    print(df_phenotypes.toPandas()['curie_id'].unique())

    # filter out bad curie rows
    df_phenotypes = df_phenotypes.where(df_phenotypes.curie_id != 'NA').where(df_phenotypes.curie_id.isNotNull())
    # df_phenotypes = df_phenotypes.where(df_phenotypes.curie_id != 'NA')
    df_phenotypes.toPandas().info()

    # get unique values
    list_unique = df_phenotypes.toPandas()['curie_id'].unique()
    print("type: {}".format(type(list_unique)))
    print(list_unique)

    # load the pValues
    df_pvalues = spark.read.csv(file_genebass, sep='\t', header=True)
    # df_pvalues.toPandas().info()
    print(df_pvalues.show(10))

    # join on the filtered phenotypes
    df_new_pvalues = df_pvalues \
        .join(df_phenotypes, 
            (df_pvalues.phenocode == df_phenotypes.phenocode) & 
            ((df_pvalues.coding == df_phenotypes.coding) | 
                (df_pvalues.coding.isNull()) & (df_phenotypes.coding.isNull())), 
            how="inner") \
        .select(df_pvalues.gene_symbol.alias('gene'), 
            df_pvalues.Pvalue_Burden.alias('p_value'), 
            df_pvalues.BETA_Burden.alias('beta'), 
            df_pvalues.SE_Burden.alias('se'), 
            df_pvalues.phenocode.alias('p_code'), 
            df_phenotypes.phenocode.alias('ph_code'), 
            df_pvalues.coding.alias('p_coding'), 
            df_phenotypes.coding.alias('ph_coding'), 
            df_phenotypes.description.alias('pheno'), 
            df_phenotypes.curie_id.alias('pheno_id'))
    df_new_pvalues.toPandas().info()
    print(df_new_pvalues.show(20))

        # .select('gene_symbol', 'phenocode', 'coding', 'Pvalue_Burden', 'BETA_Burden', 'SE_Burden', 'description', 'curie_id')
        # .join(df_phenotypes, (df_pvalues.phenocode == df_phenotypes.phenocode) & (df_pvalues.coding == df_phenotypes.coding), how="inner") \

    # write out data
    # df_new_pvalues \
    #         .write \
    #         .mode('overwrite') \
    #         .json('%s' % dir_filtered)
    # print("wrote out to directory: {}".format(dir_filtered))

    df_new_pvalues \
            .coalesce(1) \
            .write \
            .mode('overwrite') \
            .option("delimiter", "\t") \
            .option("header", "true" ) \
            .csv('%s' % dir_filtered)
    print("wrote out tsv to directory: {}".format(dir_filtered))


#   - csvDf.write.option("delimiter", "\t").csv(output_path, header='true')
#   - df.coalesce(1).write.format('json').save('myfile.json')                     # as one file


# <class 'pandas.core.frame.DataFrame'>                                           
# RangeIndex: 2901208 entries, 0 to 2901207
# Data columns (total 10 columns):
#  #   Column     Dtype 
# ---  ------     ----- 
#  0   gene       object
#  1   p_value    object
#  2   beta       object
#  3   se         object
#  4   p_code     object
#  5   ph_code    object
#  6   p_coding   object
#  7   ph_coding  object
#  8   pheno      object
#  9   pheno_id   object
# dtypes: object(10)


# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1406 entries, 0 to 1405
# Data columns (total 5 columns):
#  #   Column              Non-Null Count  Dtype 
# ---  ------              --------------  ----- 
#  0   phenocode           1406 non-null   object
#  1   coding              192 non-null    object
#  2   description         1406 non-null   object
#  3   coding_description  181 non-null    object
#  4   curie_id            1406 non-null   object
# dtypes: object(5)
# memory usage: 55.0+ KB

