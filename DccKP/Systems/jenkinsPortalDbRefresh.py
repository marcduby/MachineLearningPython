# imports
import sys
import argparse
import configparser


# method to take in secret and return tables
def show_tables(schema, username, password, host):
    '''returns the database table list from the database specified in the secret provided'''
    db = mdb.connect(host, username, password, schema)
    sql = "show tables"
    cursor = db.cursor()
    table_list = []

    # execute
    cursor.execute(sql)

    # fetch
    for row in cursor:
        table_list.append(row[0])

    # return
    return table_list

def clone_database(schema_dev, schema_new, aws_secret):
    # get the secret data
    mysql_user = aws_secret['username']
    mysql_password = aws_secret['password']
    mysql_host = aws_secret['host']

    # create the new database
    header_print("creating the new schema {}".format(schema_new))
    mysql_command_create_schema = "mysql -u {} -p'{}' -h {} -e \"create database {}\"".format(mysql_user, mysql_password, mysql_host, schema_new)
    run_system_command(mysql_command_create_schema, if_test = arg_if_test)

    # clone database
    # build the mysql schema cloning command
    header_print("copying data from schema {} to the new schema {}".format(schema_dev, schema_new))
    database_table_list = show_tables(schema_dev, mysql_user, mysql_password, mysql_host)
    for table in database_table_list:
        mysql_command_dump = "mysqldump -u {} -p'{}' -h {} {} {}".format(mysql_user, mysql_password, mysql_host, schema_dev, table)
        mysql_command_load = "mysql -u {} -p'{}' -h {} {}".format(mysql_user, mysql_password, mysql_host, schema_new)
        mysql_command_combined = mysql_command_dump + " | " + mysql_command_load
        run_system_command(mysql_command_combined, if_test = arg_if_test)

if __name__ == '__main__':
    # configure argparser
    parser = argparse.ArgumentParser("script to clone the dev portal mysql db data to the prod mysql db")
    # add the arguments
    parser.add_argument('-s', '--secret', help='the secret for the bioindex', default='dig-bio-index', required=False)
    parser.add_argument('-p', '--portal', help='the portal schema to clone', default='portal', required=False)
    parser.add_argument('-f', '--file', help='the temp .bioindex file to use', required=True)
    parser.add_argument('-t', '--test', help='if this is a dryrun/test', default=True, required=False)
    # get the args
    args = vars(parser.parse_args())

    # print the command line arguments
    header_print("printing arguments used")
    print_args(args)

    # need passed in args:
    arg_if_test = True

    # set the parameters
    if args['secret'] is not None:
        secret_name_dev = args['secret']
    if args['portal'] is not None:
        schema_portal_dev = args['portal']
    if args['file'] is not None:
        file_bioindex = str(args['directory'])
    if args['test'] is not None:
        arg_if_test = not args['test'] == 'False'

    header_print("passed AWS dev secret {} and ifTest {}".format(secret_name_dev, arg_if_test))
    header_print("using portal database {} and temp dir {}".format(schema_portal_dev, file_temp_directory))

    # get the config file used by the bioindex and parse it

    # get the database schema to refresh

    # drop all the tables in the schema 

    # clone the schema






