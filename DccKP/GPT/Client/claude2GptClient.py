

# imports
import requests
import json


# constants


# methods


# main
if __name__ == "__main__":
    pass



API_KEY = 'YOUR_API_KEY' 

API_URL = 'https://api.anthropic.com/v1/claude'

text = 'The iPhone changed the mobile phone industry when it was introduced in 2007. Developed by Apple, the iPhone featured a multi-touch screen and a user interface based on gestures. Later iPhones added features like Siri, Face ID, and powerful cameras. iPhones are seen as status symbols and allow users to access apps, music, and more in a smartphone.'

data = {
  'text': text,
  'instruction': 'Summarize this text'
}

headers = {
  'Authorization': f'Bearer {API_KEY}',
  'Content-Type': 'application/json'
}

response = requests.post(API_URL, headers=headers, data=json.dumps(data))

result = json.loads(response.text)

print(result['text'])


# claude v1
# # API key for authentication 
# API_KEY = 'YOUR_API_KEY' 

# # Endpoint for Claude API
# API_URL = 'https://api.anthropic.com/v1/claude'

# # Sample text to summarize
# text = 'The iPhone is a smartphone developed by Apple that uses iOS as its operating system. The first iPhone was released in 2007 and there have been multiple new models released since, with the latest being the iPhone 14 in 2022. iPhones are popular for their sophisticated design, easy-to-use interface, and integration with other Apple products.'

# # Data to send to API   
# data = {
#   'text': text,
#   'instruction': 'Summarize this'  
# }

# # Set request parameters
# headers = {
#   'Authorization': f'Bearer {API_KEY}',
#   'Content-Type': 'application/json'
# }

# # Make POST request and store response
# response = requests.post(API_URL, headers=headers, json=data)
# result = response.json()

# # Print summarized text
# print(result['text'])