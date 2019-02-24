
# imports
import mysql.connector
from mysql.connector import errorcode

# main variables
selectSql = "select ID from REGIONS_ABC_GENE_LOAD where PROCESSED = 'N' limit 1"


# build the chromosome list
chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
annotations = range(4, 7)

# function to enter the variants
def call_insert_variants(annotation_in, chromosome_in):
    try:
        conn1 = mysql.connector.connect(user= 'diguser', password = 'type2diabetes', host= 'db-dev-seventeen-aws.cxrzznxifeib.us-east-1.rds.amazonaws.com', database= 'digkb')
        cursor1 = conn1.cursor()

        inputs = []
        inputs.append(chromosome_in)
        inputs.append(annotation_in)
        cursor1.callproc('addVariantsToRegionTissues', inputs)

        # commit
        conn1.commit()

        # log
        print("Processed function with chromosome {} with annotation {}".format(chromosome_in, annotation_in))

    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))

    finally:
        cursor1.close()
        conn1.close()


# test call the function
call_insert_variants(1, '7')


for annot in annotations:
    for chrom in chromosomes:
        # create connection and call stored procedure
        call_insert_variants(annot, chrom)

        # log
        print("Processed chromosome {} with annotation {}".format(chrom, annot))


