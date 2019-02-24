
# imports
import mysql.connector
from mysql.connector import errorcode

# main variables
selectSql = "select ID from REGIONS_ABC_GENE_LOAD where PROCESSED = 'N' limit 1"

mysqlCreds = {'user': 'diguser', 'password': 'type2diabetes', 'host': 'db-dev-seventeen-aws.cxrzznxifeib.us-east-1.rds.amazonaws.com', 'database': 'digkb'}

def get_gene():
    try:
        conn = mysql.connector.connect(user= 'diguser', password = 'type2diabetes', host= 'db-dev-seventeen-aws.cxrzznxifeib.us-east-1.rds.amazonaws.com', database= 'digkb')
        cursor = conn.cursor(buffered = True)

        cursor.execute(selectSql)

        result_args = cursor.fetchone()


    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))

    finally:
        cursor.close()
        conn.close()

    return result_args[0]


def call_insert_variants(inputGene):
    try:
        conn1 = mysql.connector.connect(user= 'diguser', password = 'type2diabetes', host= 'db-dev-seventeen-aws.cxrzznxifeib.us-east-1.rds.amazonaws.com', database= 'digkb')
        cursor1 = conn1.cursor()

        gene = []
        gene.append(inputGene)
        cursor1.callproc('addVariantsToAbc', gene)

        # commit
        conn1.commit()

    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))

    finally:
        cursor1.close()
        conn1.close()


for num in range(1, 5):
    # get the gene
    gene = get_gene()
    print "{} - {}".format(num, gene)

    # call the variant procedure for the gene
    call_insert_variants(gene)







