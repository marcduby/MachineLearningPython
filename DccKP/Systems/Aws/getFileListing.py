
# using subprocess to avoid having to install boto3 on spun up instance

# imports
import subprocess
import boto3 

# get a directory listing
# result = subprocess.run(['aws', 's3', 'ls', 's3://dig-analysis-data/out/finemapping/variant-associations/AF/ancestry=EA', '--recursive', '--human-readable', '--summarize'])
result = subprocess.run(['aws', 's3', 'ls', 's3://dig-analysis-data/out/finemapping/variant-associations/AF/', '--human-readable', '--summarize'], stdout=subprocess.PIPE)
print("got result type {} and output {}".format(type(result), result))
print("got result stdout type {} and output {}".format(type(result.stdout), result.stdout))


# usign boto3
s3 = boto3.client("s3")
all_objects = s3.list_objects(Bucket = 'dig-analysis-data/out/finemapping/variant-associations/AF')
print("got result type {} and output {}".format(type(all_objects), all_objects))
