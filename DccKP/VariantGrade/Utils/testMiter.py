
# imports
import requests

# constants
url_service_miter = "http://miter.broadinstitute.org/mitergrade/?query=p.P12A&prevalence=1.0e-5"
url_service_mutantp53 = "http://mutantp53.broadinstitute.org/mutantp53grade/?query=p.R273H&prevalence=1.0e-5"
url_service_kras = "http://krasgrade.genetaics.com/krasgrade/?query=p.G13P&prevalence=1.0e-5"

# tests
def query_api(url, debug=False):
    ''' 
    method to test the getg query of the server 
    '''

    # call the query service
    response = requests.get(url)

    # test
    if debug:
        print("the status code is: {}".format(response.status_code))
    assert response.status_code == 200

def test_miter(debug=False):
    query_api(url = url_service_miter, debug=debug)

def test_mutantp53(debug=False):
    query_api(url = url_service_mutantp53, debug=debug)

def test_kras(debug=False):
    query_api(url = url_service_kras, debug=debug)

if __name__ == "__main__":
    # test the services
    test_miter(True)

    test_mutantp53(True)

    test_kras(True)