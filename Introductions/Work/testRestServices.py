
# import
from urllib import request
from urllib import parse
import sys

# method to call a rest service
def call_rest(url, params):
    """method to call REST service, checking the error code"""
    error_code = -10

    # build the GET string
    target_url = url
    if bool(params):
        target_url += "?" + parse.urlencode(params)

    # print
    # print("target url is: {}".format(target_url))

    try:
        # make the call
        response = request.urlopen(target_url)

        # get the error code
        error_code = response.code
    except:
        print("Got excpeption")

    # return the error code
    return error_code

def build_rest(host, path):
    """method to build the REST path for the call"""
    return host + "/" + path

# script values
test_host = "http://localhost:8090/dccservices"

if len(sys.argv) > 1 and bool(sys.argv[1]):
    test_host = sys.argv[1]

# dictionary of rest paths to params
rest_path_hash = {
    'getAggregatedDataSimple': {"phenotype": "T2D", "chrom": "3", "start": 12568817, "end": 12575905},
    'graph/tissue/list/object': {}}

# test with public KB
public_kb = "http://public.type2diabeteskb.org/dccservices/getAggregatedDataSimple"
public_params = {"phenotype": "T2D", "chrom": "3", "start": 12228817, "end": 12575905}
code = call_rest(public_kb, public_params)

print("the returned code was: {}".format(code))


# test with loop
print("\ntesting collection of REST calls")
for key, value in rest_path_hash.items():
    test_url = build_rest(test_host, key)
    code = call_rest(test_url, value)
    test_message = '\nFAILURE'

    # test code
    if code == 200:
        test_message = 'SUCCESS'

    # print
    print("{} - code {} for testing {}".format(test_message, code, test_url))


