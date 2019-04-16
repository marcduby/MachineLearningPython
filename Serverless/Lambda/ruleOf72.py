import json
from decimal import Decimal


def lambda_handler(event, context):
    amount = float(100)
    interest = 7.2
    years = 10

    json_data = event["queryStringParameters"]

    # set the values
    input_amount = float(str(json_data['amount']))
    input_interest = float(str(json_data['interest']))
    input_years = int(str(json_data['years']))

    amount = input_amount
    for d in range(input_years):
        amount = amount + amount * (input_interest / float(100))

    # build the return
    message = 'The amount ${:,.2f} compounded for {} years at {}% interest is: ${:,.2f}'.format(round(input_amount, 2),
                                                                                                input_years,
                                                                                                input_interest,
                                                                                                round(amount, 2))

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps(message)
    }
