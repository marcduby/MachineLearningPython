

#@ Imports
import requests
import json
import argparse 

# constants
JSON_INPUT_FILE = "./parkinsonCrohns.json"

# methods
def submit_to_ars(m):
    submit_url = 'https://ars.transltr.io/ars/api/submit'
    response = requests.post(submit_url,json=m)
    try:
        message_id = response.json()['pk']
    except:
        print('fail')
        message_id = None
    print(f'https://arax.ncats.io/?source=ARS&id={message_id}')
    return message_id


def retrieve_ars_results(mid):
    message_url = f'https://ars.transltr.io/ars/api/messages/{mid}?trace=y'
    response = requests.get(message_url)
    j = response.json()
    print( j['status'] )
    results = {}
    for child in j['children']:
        if child['status'] == 'Done':
            childmessage_id = child['message']
            child_url = f'https://ars.transltr.io/ars/api/messages/{childmessage_id}'
            child_response = requests.get(child_url).json()
            try:
                nresults = len(child_response['fields']['data']['message']['results'])
                if nresults > 0:
                    results[child['actor']['agent']] = {'message':child_response['fields']['data']['message']}
            except:
                nresults=0
        else:
            nresults = 0
        print( child['status'], child['actor']['agent'],nresults )
    return results

def print_args(arg_map):
    for key in arg_map.keys():
        print("   {} ===> {}".format(key, arg_map[key]))


# main
if __name__ == "__main__":
    # configure argparser
    parser = argparse.ArgumentParser("script to clone the dev bioindex data to the prod machine")
    # add the arguments
    parser.add_argument('-i', '--inputFile', help='the input json file', default=JSON_INPUT_FILE, required=True)

    # get the args
    args = vars(parser.parse_args())

    # print the command line arguments
    print_args(args)

    # set the parameters
    if args['inputFile'] is not None:
        file_input = args['inputFile']

    # read json payload
    with open(file_input) as file_json:
        json_input = json.load(file_json)

    # log
    print("calling ARS with json payload: \n{}".format(json.dumps(json_input, indent=2)))

    # submit
    key_ars = submit_to_ars(json_input)

    # print the ars query key
    print("got ARS response key: {}".format(key_ars))