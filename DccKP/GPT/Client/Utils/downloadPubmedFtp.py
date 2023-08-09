
# imports
from ftplib import FTP
import os
import requests
from bs4 import BeautifulSoup


# constants
DIR_LOCAL = "/scratch/Javaprog/Data/Broad/GPT/Pubmed"
URL_PUBMED_BASELINE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
DIR_PUBMED = "/pubmed/baseline/"

# methods

# main
if __name__ == "__main__":
    # initialize
    log = True
    list_files = []

    # get the connection
    response = requests.get(URL_PUBMED_BASELINE)
    if log:
        print("got response: {}".format(response.text))

    # get all the links
    soup = BeautifulSoup(response.text, 'html.parser')
    for link in soup.find_all('a'):
        ref = link.get('href')
        if ref.endswith('.gz'): 
            list_files.append(ref)
            print(ref)        

    # files = response.text.split('\n')
    # if log:
    #     print("for file list: {}".format(files[10]))

    for file in list_files:
        if file.endswith('.gz'): 
            # get the file
            print("saving file: {}".format(file))
            file_url = URL_PUBMED_BASELINE + file
            r = requests.get(file_url)

            # save the file
            with open(os.path.join(DIR_LOCAL, file), 'wb') as f:
                f.write(r.content)
            


# import requests 
# from bs4 import BeautifulSoup

# url = 'http://example.com/'

# response = requests.get(url)

# soup = BeautifulSoup(response.text, 'html.parser')

# for link in soup.find_all('a'):
#     print(link.get('href'))


# from claude
# from ftplib import FTP
# import os

# ftp = FTP('ftp.example.com') 
# ftp.login(user='username', passwd = 'password')
# ftp.cwd('directory') 

# for filename in ftp.nlst():
#     if filename.endswith('.txt'): 
#         local_filename = os.path.join('/local/path', filename)
#         file = open(local_filename, 'wb')
#         ftp.retrbinary('RETR ' + filename, file.write)
#         file.close()
        
# ftp.quit()


# import requests
# import os

# url = 'http://example.com/files/'

# response = requests.get(url)
# files = response.text.split('\n')

# for file in files:
#     if file.endswith('.txt'):
#         file_url = url + file
#         r = requests.get(file_url)
#         with open(os.path.join('/local/path', file), 'wb') as f:
#             f.write(r.content)


