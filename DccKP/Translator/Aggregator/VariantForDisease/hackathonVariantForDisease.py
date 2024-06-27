

# imports
import requests
import os
import json

# constants
URL_DISEASE_GENE = "https://bioindex-dev.hugeamp.org/api/bio/query/gene-associations?q={}"
URL_VARIANTS_DISEASE_REGION = "https://bioindex-dev.hugeamp.org/api/bio/query/associations?q={},{}:{}-{}"
URL_PHENOTYPES = "https://bioindex-dev.hugeamp.org/api/portal/phenotypes"
GENE_REGIONS = []


# methods
def read_in_phenotypes(file_path='phenotypes.json', log=False):
    '''
    reads in the phenotype file 
    '''
    map_phenotypes = {}

    with open(file_path, 'r') as file:
        # Step 3: Parse the JSON data
        json_data = json.load(file)

        for row in json_data.get('data'):
            map_phenotypes[row.get('name')] = row.get('description')

    # return
    return map_phenotypes


def rest_get_disease_for_gene(gene, num_top=10, log=False):
    '''
    gets the top n disease for a gene 
    '''
    list_result = []
    chrom = None
    start = None
    end = None

    # get the phenotypes
    response = requests.get(URL_DISEASE_GENE.format(gene)).json()
    for row in response.get('data'):
        list_result.append({'phenotype_code': row.get('phenotype'), 'p_value': row.get('pValue')})
        chrom = row.get('chromosome')
        start = row.get('start')
        end = row.get('end')

    # return the top n
    return list_result[:num_top], chrom, start, end


def rest_get_variants_for_region_phenotype(chrom, start, end, gene, phenotype, num_top=20, list_annotation=['missense_variant'], log=False):
    '''
    returns the misense vars for a region/phenotype, filtering by var type 
    '''
    list_result = []

    # get the data
    url = URL_VARIANTS_DISEASE_REGION.format(phenotype, chrom, start, end)
    if log:
        print("using url: {}".format(url))
    response = requests.get(url).json()

    # parse
    list_data = response.get('data')
    for row in list_data:
        if row.get('consequence') in list_annotation:
            if row.get('nearest') and gene in row.get('nearest'):
                list_result.append({'phenotype': phenotype, 'gene': gene, 'variant': row.get('varId'), 'p_value': row.get('pValue'), 'annotation': row.get('consequence')})

    # return
    return list_result


# main
if __name__ == "__main__":
    # initialize 
    gene = 'PPARG'
    gene = 'TP53'
    num = 5
    chrom = None
    start = None
    end = None

    # read in the phenotypes
    map_phenotypes = read_in_phenotypes()

    # get the diseases for the gene
    list_gene_phenotypes, chrom, start, end = rest_get_disease_for_gene(gene=gene, num_top=num)

    # print
    for row in list_gene_phenotypes:
        print("for gene: {} (chrom: {}, start: {}, end: {}), got top {} disease: '{}' with pValue: {}".format(gene, 
            chrom, start, end, num, map_phenotypes.get(row.get('phenotype_code')), row.get('p_value')))

        # for each phenotype, find the variants
        list_variants = rest_get_variants_for_region_phenotype(chrom=chrom, start=start, end=end, gene=gene, phenotype=row.get('phenotype_code'))
        print("got list of variants:\n{}".format(json.dumps(list_variants, indent=2)))