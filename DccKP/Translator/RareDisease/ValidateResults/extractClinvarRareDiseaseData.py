
# imports
import csv
from collections import defaultdict
import json
import requests

# constants
DIR_DATA = "/home/javaprog/Data/Broad/RareDisease"
# phenotype data below from https://www.ncbi.nlm.nih.gov/clinvar/submitters/505999/
FILE_DATA = "{}/clinvarRareDisease_result.txt".format(DIR_DATA)
URL_RARE_DISEASE = "https://translator.broadinstitute.org/genetics_provider/rare_disease_calc/gene_scores"
FILE_OUT_RESULTS = "{}/rareDiseaseScores.json".format(DIR_DATA)


# methods
def extract_gene_phenotypes(filename):
    """
    Parse the tab-delimited clinical variant file and extract:
        Gene → list of associated phenotypes (Condition(s))
    """
    gene_to_pheno = defaultdict(set)

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            genes_field = row.get("Gene(s)", "").strip()
            pheno_field = row.get("Condition(s)", "").strip()

            # Skip rows without gene information
            if not genes_field:
                continue

            # split multi-gene rows: "A|B|C"
            genes = [g.strip() for g in genes_field.split("|") if g.strip()]

            # Some rows may have empty condition(s)
            if pheno_field:
                phenotypes = [p.strip() for p in pheno_field.split("|") if p.strip()]
            else:
                phenotypes = []

            # Attach phenotypes to each gene
            for g in genes:
                for p in phenotypes:
                    gene_to_pheno[g].add(p)

    # Convert sets to sorted lists
    return {gene: sorted(list(phenos)) for gene, phenos in gene_to_pheno.items()}


def filter_genes_by_min_phenotypes(gene_to_pheno, min_count=3):
    """
    Return a new dict containing only genes whose phenotype list
    has at least `min_count` elements (default = 3).
    """
    return {
        gene: phenos
        for gene, phenos in gene_to_pheno.items()
        if len(phenos) >= min_count
    }


def clean_phenotype_list(phenotypes):
    """
    Remove 'not provided' and 'not specified' (case-insensitive)
    from a list of phenotype strings.
    Returns a cleaned list.
    """
    remove_set = {"not provided", "not specified", 'see cases'}
    
    cleaned = [
        p for p in phenotypes
        if p.lower().strip() not in remove_set
    ]
    return cleaned


def filter_out_gene_name_in_phenotypes(gene_to_pheno):
    """
    Remove map entries where any phenotype string contains the gene name.
    Matching is case-insensitive and substring-based.
    
    Example:
        EP300 → ["Atypical Rubinstein-Taybi", "EP300-related disorder"]
        → this entry will be removed
    """
    filtered = {}

    for gene, phenos in gene_to_pheno.items():
        # gene_lower = gene.lower()

        # # If any phenotype contains the gene name, skip this gene entirely
        # if any(gene_lower in p.lower() for p in phenos):
        #     continue

        # If any phenotype contains the gene name, skip this gene entirely
        if any(gene in p for p in phenos):
            continue

        filtered[gene] = phenos

    return filtered


def summarize_gene_position(avg_genes, input_gene):
    """
    avg_genes: list of dicts like:
        { "avg_score": float, "gene": "ABC" }

    input_gene: gene symbol to search for.

    Returns a dict containing:
        - index of input gene (or -1)
        - percentage of gene score relative to max
        - max score
        - min score
    """

    if not avg_genes:
        return {
            "index": -1,
            "pct_of_max": 0.0,
            "max_score": 0.0,
            "min_score": 0.0
        }

    # Extract all scores
    scores = [entry["avg_score"] for entry in avg_genes]
    max_score = max(scores)
    min_score = min(scores)
    avg_score = sum(scores) / len(scores)

    # Find gene index and score if present
    index = -1
    gene_score = None
    for i, entry in enumerate(avg_genes):
        if entry["gene"].upper() == input_gene.upper():
            index = i
            gene_score = entry["avg_score"]
            break

    # If gene not found
    if index == -1:
        return {
            "index": -1,
            "pct_of_max": 0.0,
            "max_score": max_score,
            "min_score": min_score
        }

    # Compute percentage relative to max
    pct = (gene_score / max_score) * 100 if max_score != 0 else 0.0

    return {
        "index": index,
        "pct_of_max": pct,
        "max_score": max_score,
        "min_score": min_score,
        "avg_score": avg_score,
        "gene_score": gene_score
    }


def get_map_keys_subset(map_input, count=3):
    '''
    returns a subset of the keys
    '''
    list_keys = list(map_input.keys())

    return list_keys[:count]


def query_phenotypes(phenotypes, url=URL_RARE_DISEASE, timeout=200):
    """
    Send a POST request to the given URL with a list of phenotype strings.
    
    Parameters:
        url (str): target REST endpoint
        phenotypes (list[str]): list of phenotype IDs or names
        timeout (int): request timeout (default 20 seconds)
    
    Returns:
        dict: parsed JSON response
    """

    # Prepare JSON payload
    payload = {"phenotypes": phenotypes}

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()  # raise error for 4xx/5xx
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"REST call failed: {e}")
        return {"error": str(e)}
    


def write_out_json(json_result, file_out=FILE_OUT_RESULTS):
    '''
    wites out json file
    '''
    with open(file_out, 'w') as f:
        json.dump(json_result, f, indent=2)



if __name__ == "__main__":
    input_file = FILE_DATA  # ← your file
    result = extract_gene_phenotypes(input_file)

    # Print results
    for gene, phenos in result.items():
        print(f"{gene}: {phenos}")

    # remove the unspecified entries
    map_filtered = {}
    for key, value in result.items():
        map_filtered[key] = clean_phenotype_list(value)

    # now filter genes where a phenotype has the name in them
    map_filtered = filter_out_gene_name_in_phenotypes(gene_to_pheno=map_filtered)

    # now only keep the lists with at least 3 phenotypes
    map_multiple_phenotypes = filter_genes_by_min_phenotypes(gene_to_pheno=map_filtered, min_count=5)

    # print
    print("\n\nfila::\n:{}".format(json.dumps(map_multiple_phenotypes, indent=2)))
    print("gene list if size: {}".format(len(map_multiple_phenotypes)))

    # get the subset of the genes
    list_genes = get_map_keys_subset(map_input=map_multiple_phenotypes, count=3)

    # for each gene, query the rest service
    map_gene_scores = {}
    for gene in list_genes:
        list_phenotypes = map_multiple_phenotypes.get(gene, [])
        # log
        print("querying REST for gene: {} and phenotypes: {}\n".format(gene, json.dumps(list_phenotypes)))

        # get the REST results
        json_for_gene = query_phenotypes(phenotypes=list_phenotypes)

        # get the result scores for the gene
        map_score_for_gene = summarize_gene_position(avg_genes=json_for_gene.get('output', {}).get('avg_genes', []), input_gene=gene)
        map_score_for_gene['phenotypes'] = list_phenotypes
        map_gene_scores[gene] = map_score_for_gene

    # print
    print("got final score sheet: \n{}".format(json.dumps(map_gene_scores, indent=2)))
    write_out_json(json_result=map_gene_scores)