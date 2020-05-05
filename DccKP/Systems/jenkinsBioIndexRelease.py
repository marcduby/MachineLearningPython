# imports
import os
import time
import datetime
import json
import time
import sys

import boto3
import base64
from botocore.exceptions import ClientError


# script variables
# get the timestamp for the build
format = "%Y%m%d%H%M%S"
timestamp = time.strftime(format)

print("time stamp is: {}".format(timestamp))

# DB settings
schema_name_dev = "biodev"
schema_name_new = 'bioindex_' + timestamp

# s3 settings
s3_bucket_new = 'dig-bio-index-' + timestamp
s3_bucket_dev = 'dig-bio-index-dev'

# git settings
code_directory = '/Users/mduby/BioIndex/'
git_directory = code_directory + 'bioindex_' + timestamp
git_clone_command = "git clone git@github.com:broadinstitute/dig-bioindex.git " + git_directory

# secrets settings
secret_name_dev = "bioindex-dev"
secret_name_new = "bioindex-" + timestamp
region_name = "us-east-1"

# get the aws client and session
s3client = boto3.client('s3')


# log
# print("got initial schema name: {} and s3 index: {} and git directory {}".format(schema_name_dev, s3_bucket_dev, git_directory))
# print("the git clone command is: {}".format(git_clone_command))

# methods
# method to run an OS command and time it
def run_system_command(os_command, input_message = "", if_test = True):
    log_message = "Running command"
    exit_code = None
    start = time.time()
    if if_test:
        log_message = "Testing command"
    print("{}: {}".format(log_message, os_command))
    if not if_test:
        exit_code = os.system(os_command)
    end = time.time()
    print("    Done in {}s with exit code {}".format(end - start, exit_code))

def header_print(message):
    print("\n==> {}".format(message))

# method to list the buckets based on search string
def print_s3_buckets(s3client, search_str):
    # print all the bucket names
    list_buckets_resp = s3client.list_buckets()
    for bucket in list_buckets_resp['Buckets']:
        if search_str in bucket['Name']:
            print("existing bucket name after addition: {}".format(bucket['Name']))


# method to retrive the secrets given name and region
def get_secret(secret_name, region_name):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])

    return json.loads(secret)


if __name__ == "__main__":
    # need passed in args:
    arg_if_test = True

    # get the command line arguments
    if (sys.argv) and len(sys.argv) > 3:
        secret_name_dev = sys.argv[1]
        s3_bucket_dev = sys.argv[2]
        schema_name_dev = sys.argv[3]
        if len(sys.argv) > 4:
            arg_if_test = sys.argv[4]
        print("usiing secret '{}' and s3 bucket '{}' and schema'{}' and isTest '{}'".format(secret_name_dev, s3_bucket_dev, schema_name_dev, arg_if_test))
    else:
        print("Usage: python3 jenkinsBioIndexRelease.py <secret> <s3_bucket> <schema_name> <dry_run>")
        exit()

    header_print("passed in bucket is {} AWS dev secret {} and ifTest {}".format(s3_bucket_dev, secret_name_dev, arg_if_test))

    # clone the code (not needed anymore since config.json not used for indexes)
    # log
    # print("Running git clone command: {}".format(git_clone_command))

    # # clone the bioindex
    # os.chdir(code_directory)
    # clone_output = os.system(git_clone_command)

    # # log
    # print("the git process exited with code {}".format(clone_output))

    # get the secret to use to clone
    header_print("get the secret to clone")
    bio_secret_dev = get_secret(secret_name_dev, region_name)
    print("got secret with name {}".format(bio_secret_dev['dbInstanceIdentifier']))

    # list the existing buckets before creating the new one
    header_print("listing existing s3 buckets")
    print_s3_buckets(s3client, 'index')

    # create the new s3 busket
    header_print("creating the new s3 bucket")
    # create the s3 bucket
    if not arg_if_test:
        s3client.create_bucket(Bucket=s3_bucket_new)
        print("created new s3 bucket {}".format(s3_bucket_new))
    else:
        print("test, so skipped creating new s3 bucket {}".format(s3_bucket_new))

    list_buckets_resp = s3client.list_buckets()
    for bucket in list_buckets_resp['Buckets']:
        if bucket['Name'] == s3_bucket_new:
            print('(Just created) --> {} - there since {}'.format(bucket['Name'], bucket['CreationDate']))

    # list the existing buckets before creating the new one
    header_print("listing existing s3 buckets")
    print_s3_buckets(s3client, 'index')

    # sync the new s3 buckeet with the data from the given s3 bucket
    header_print("sub folders of {} that need to be cloned".format(s3_bucket_dev))
    result = s3client.list_objects(Bucket=s3_bucket_dev, Prefix="", Delimiter='/')
    for s3object in result.get('CommonPrefixes'):
        print("-> sub folder: {}".format(s3object.get('Prefix')))

    # log
    header_print("cloning s3 bucket {}".format(s3_bucket_dev))
    # copy the data
    for s3object in result.get('CommonPrefixes'):
        s3_subdirectory = s3object.get('Prefix')
        s3_command = "aws s3 sync s3://{}/{} s3://{}/{}".format(s3_bucket_dev, s3_subdirectory, s3_bucket_new, s3_subdirectory)
        run_system_command(s3_command, if_test = arg_if_test)

    # get the db parameters
    # build the mysql command
    mysql_user = bio_secret_dev['username']
    mysql_password = bio_secret_dev['password']
    mysql_host = bio_secret_dev['host']
    schema_name_dev = bio_secret_dev['dbname']

    # build the create the database
    header_print("creating the new schema {}".format(schema_name_new))
    mysql_command_create_schema = "mysql -u {} -p'{}' -h {} -e \"create database {}\"".format(mysql_user, mysql_password, mysql_host, schema_name_new)
    run_system_command(mysql_command_create_schema, if_test = arg_if_test)

    # clone database
    # build the mysql schema cloning command
    header_print("copying data from schema {} to the new schema {}".format(schema_name_dev, schema_name_new))
    mysql_command_dump = "mysqldump -u {} -p'{}' -h {} {}".format(mysql_user, mysql_password, mysql_host, schema_name_dev)
    mysql_command_load = "mysql -u {} -p'{}' -h {} {}".format(mysql_user, mysql_password, mysql_host, schema_name_new)
    mysql_command_combined = mysql_command_dump + " | " + mysql_command_load
    run_system_command(mysql_command_combined, if_test = arg_if_test)

    # create the new secret and add in the parameters
    header_print("creating new AWS secret {}".format(secret_name_new))
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    if not arg_if_test:
        client.create_secret(Name = secret_name_new)
        print("created new secret {}".format(secret_name_new))
    else:
        print("test, so skipped creating new secret {}".format(secret_name_new))


    # populate the secret with data
    header_print("updating AWS secret {} with new dict values".format(secret_name_new))
    new_secret_dict = {'username': mysql_user, 'password': mysql_password, 'host': mysql_host, \
        'dbname': schema_name_new, 's3bucket': s3_bucket_new, 'engine': 'mysql', 'port': 3306}
    if not arg_if_test:
        client.put_secret_value(SecretId = secret_name_new, SecretString = json.dumps(new_secret_dict))
        print("updated new secret {}".format(secret_name_new))
    else:
        print("test, so skipped updating new secret {}".format(secret_name_new))

    # 
    # log done
    header_print("DONE\n\n\n")


    # # testing command 
    # header_print("DEBIG")
    # testBoolean = False
    # print("testing the False arg ifTest with {}".format(testBoolean))
    # run_system_command("junk command", if_test = testBoolean)

