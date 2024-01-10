
# imports
import csv
import pymysql as mdb
import os 

# constants
SCHEMA_PHE = "phe_disease"
FILE_LOCATION = "/scratch/Javaprog/Data/Broad/DiseaseToPhenotype/phenotype.hpoa"
DB_PASSWD = os.environ.get('DB_PASSWD')

# SQL
SELECT_DISEASE = "select id from phen_disease where curie = %s"
INSERT_DISEASE = "insert into phen_disease (curie, name) values(%s, %s)"
INSERT_DISEASE_LINK = "insert into phen_phenotype_disease_link (disease_curie, phenotype_curie) values(%s, %s)"

# methods
def get_connection(schema=SCHEMA_PHE):
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=schema)

    # return
    return conn

def get_db_disease(cursor, curie, log=False):
    '''
    looks for a disease by curie
    '''
    # initialize
    result_id = None

    # look up
    cursor.execute(SELECT_DISEASE, (curie))
    db_result = cursor.fetchall()
    if db_result:
        result_id = db_result[0][0]

    # return
    return result_id


def insert_db_disease(cursor, curie, name, log=False):
    '''
    inserts a disease into the db
    '''
    # initialize
    temp_id = None

    # look for the disease
    temp_id = get_db_disease(cursor=cursor, curie=curie, log=log)

    # if no, insert
    if not temp_id:
        cursor.execute(INSERT_DISEASE, (curie, name))
        print("inserted disease: {} with name: {}".format(curie, name))


def insert_db_phenotype_link(conn, disease_curie, disease_name, phenotype_curie, log=False):
    '''
    insert a phenotype/disease link in the db
    '''
    # initialize
    temp_id = None
    cursor = conn.cursor()

    # insert the disease
    insert_db_disease(cursor=cursor, curie=disease_curie, name=disease_name, log=log)

    # insert the phenotype link
    cursor.execute(INSERT_DISEASE_LINK, (disease_curie, phenotype_curie))
    print("inserted link to disease: {} with phenotype: {}".format(disease_curie, phenotype_curie))


# main
if __name__ == "__main__":
    # initialize
    results = []

    # read the data
    with open(FILE_LOCATION) as f:
        tsv_file = csv.reader(f, delimiter='\t')

        # Skip first 4 lines 
        for _ in range(4):
            next(tsv_file)

        # Read remaining lines        
        for line in tsv_file:
            results.append({
                'disease_id': line[0], 
                'disease_name': line[1], 
                'phenotype_id': line[3]
            })

    print(results)

    # get db connection
    conn = get_connection()

    # insert disease into db if necessary
    for row in results:
        insert_db_phenotype_link(conn=conn, disease_curie=row.get('disease_id'), disease_name=row.get('disease_name'), phenotype_curie=row.get('phenotype_id'))
        conn.commit()


