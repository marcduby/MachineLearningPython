
# data
# http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=1&bgcolor=FFFEE8
# http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=2&bgcolor=FFFEE8
# http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=3&bgcolor=FFFEE8

# reference code from video https://www.youtube.com/watch?v=15f4JhJ8SiQ&t=505s

# imports
import pandas as pd 
import requests 
from bs4 import BeautifulSoup
import json 

# constants
url_type01 = 'http://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Cell+Line&tier=1&bgcolor=FFFEE8'
file_bassett_labels = "/home/javaprog/Data/OldNUC/Broad/Basset/Production/basset_labels.txt"
file_bassett_labels_out = "/home/javaprog/Data/OldNUC/Broad/Basset/Production/basset_labels_with_text.json"

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

def get_tissue_map_data(url):
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
            new_tissue = {'cellCode': row.find_all('td')[0].text, 
                            'cellDescription': row.find_all('td')[2].text,
                            'lineage': row.find_all('td')[3].text.strip() if len(row.find_all('td')[3].text.strip()) > 0 else None, 
                            'tissue': row.find_all('td')[4].text.strip() if len(row.find_all('td')[4].text.strip()) > 0 else None, 
                            'karyotype': row.find_all('td')[5].text.strip() if len(row.find_all('td')[5].text.strip()) > 0 else None}
            
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
    array_tissues += get_tissue_map_data(url)
    print("the tissue array is of size {}".format(len(array_tissues)))

# create a pandas dataframe
df_tissues = pd.DataFrame(array_tissues, columns=['cellCode', 'cellDescription', 'lineage', 'tissue', 'karyotype'])
print(df_tissues.info())
print(df_tissues.head(10))

# read the data file
df_bassett_labels = pd.read_csv(file_bassett_labels, names=['cellCode'])

# print
print()
print(df_bassett_labels.info())
print()
print(df_bassett_labels.head(10))
list_labels = df_bassett_labels['cellCode'].tolist()
print("got labels list of size {}".format(len(list_labels)))

# # inner join the two cell label data 
# df_joined = pd.merge(df_bassett_labels, df_tissues, on='cell_code')
# print()
# print(df_joined.info())
# print()
# print(df_joined.head(10))

# create new array of tissues
list_final_tissues = [tissue for tissue in array_tissues if tissue.get('cellCode') in list_labels]
print()
print("final size of joined labels is {}".format(len(list_final_tissues)))

# write out the results
with open(file_bassett_labels_out, 'w') as out_file:
    for row in list_final_tissues:
        out = json.dumps(row, separators=(',', ':'))
        out_file.write(out)
        out_file.write('\n')

