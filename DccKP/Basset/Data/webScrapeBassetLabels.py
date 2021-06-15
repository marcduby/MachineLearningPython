
# data
# http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=1&bgcolor=FFFEE8
# http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=2&bgcolor=FFFEE8
# http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=3&bgcolor=FFFEE8


# imports
import pandas as pd 
import requests 
from bs4 import BeautifulSoup

# constants
url_type01 = 'http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=1&bgcolor=FFFEE8'
file_bassett_labels = "/home/javaprog/Data/OldNUC/Broad/Basset/Production/basset_labels.txt"

def get_tissue_array_data(url):
    ''' takes in url, returns tissue array df of results of table '''
    # initialize
    results = []

    # get the request
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')

    # get data
    table_data = soup.find('table', class_='sortable')
    print()
    # print(table_data)

    for body in table_data.find_all('tbody'):
        rows = body.find_all('tr')
        for row in rows:
            new_tissue = [row.find_all('td')[0].text, 
                            row.find_all('td')[2].text,
                            row.find_all('td')[3].text, 
                            row.find_all('td')[4].text, 
                            row.find_all('td')[5].text]
            
            # print(new_tissue)
            results.append(new_tissue)

    # return
    print("the temp data is of size {}".format(len(results)))
    return results

if __name__ == '__main__':
    # initialize
    array_tissues = []

    # build the url array
    array_url = [
        'http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=1&bgcolor=FFFEE8',
        'http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=2&bgcolor=FFFEE8',
        'http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=3&bgcolor=FFFEE8'
    ]

# loop
for url in array_url:
    array_tissues += get_tissue_array_data(url)
    print("the tissue array is of size {}".format(len(array_tissues)))

# create a pandas dataframe
df_tissues = pd.DataFrame(array_tissues, columns=['cell_code', 'cell_description', 'lineage', 'tissue', 'karyotype'])
print(df_tissues.info())
print(df_tissues.head(10))

# read the data file
df_bassett_labels = pd.read_csv(file_bassett_labels, names=['cell_code'])

# print
print()
print(df_bassett_labels.info())
print()
print(df_bassett_labels.head(10))

# inner join the two cell label data 
df_joined = pd.merge(df_bassett_labels, df_tissues, on='cell_code')
print()
print(df_joined.info())
print()
print(df_joined.head(10))

