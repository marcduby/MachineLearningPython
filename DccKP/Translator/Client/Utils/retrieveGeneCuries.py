from Bio import Entrez

# Function to fetch NCBI Gene ID using gene symbol
def fetch_gene_id(gene_symbol):
    Entrez.email = "your-email@example.com"  # Always tell NCBI who you are
    search_handle = Entrez.esearch(db="gene", term=gene_symbol)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    
    # If a result is found, return the first gene ID
    if search_results['IdList']:
        return search_results['IdList'][0]
    else:
        return None

# List of gene symbols
genes = ["ABCA1", "ABCA7", "ABCG1", "ABPP", "ACE"]

# Dictionary to store gene symbols and their corresponding NCBI Gene IDs
gene_ids = {gene: fetch_gene_id(gene) for gene in genes}

# Print the results
for gene, gene_id in gene_ids.items():
    print(f"{gene}: {gene_id}")


