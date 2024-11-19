



from Bio import Entrez

# Function to fetch gene IDs associated with a specific term
def fetch_gene_ids(term, retmax=50):
    Entrez.email = "haroldduby@gmail.com"  # Always tell NCBI who you are
    search_handle = Entrez.esearch(db="gene", term=term, retmax=retmax)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    return search_results['IdList']

# Search term
search_term = "depression"

# Fetch top 50 gene IDs associated with the term "depression"
gene_ids = fetch_gene_ids(search_term, retmax=50)

# Print the list of gene IDs
print(gene_ids)





