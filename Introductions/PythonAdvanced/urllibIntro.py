
# import
from urllib import request
from urllib import parse

# dir
print(dir(request))

# make a request
# resp = request.urlopen("http://www.google.com")
resp = request.urlopen("http://www.wikipedia.org")

# inspect response
print()
print(type(resp))
print()
print(dir(resp))

# get the response code
print()
print("the response code was: {}".format(resp.code))
print("the response size was: {}".format(resp.length))
print("response peek: {}".format(resp.peek()))
print("the response closed status is: {}".format(resp.isclosed()))

# get the data
data = resp.read()
print()
print("the data is of type: {}".format(type(data)))
print("the data size is: {}".format(len(data)))
print("the response closed status is: {}".format(resp.isclosed()))

# get the html
html = data.decode("UTF-8")
print()
print("the html type is: {}".format(type(html)))
print("the html is: {}".format(html[:500]))


# make get request
params = {"v": "dude", "t": "5m56s"}
querystring = parse.urlencode(params)
print()
print("the query string is: {}".format(querystring))




