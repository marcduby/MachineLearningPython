

# imports
import requests
import os
import json
import csv

# constants
URL_DISEASE_GENE = "https://bioindex-dev.hugeamp.org/api/bio/query/gene-associations?q={}"
URL_VARIANTS_DISEASE_REGION = "https://bioindex-dev.hugeamp.org/api/bio/query/associations?q={},{}:{}-{}"
URL_PHENOTYPES = "https://bioindex-dev.hugeamp.org/api/portal/phenotypes"
GENE_REGIONS = []

PATH_FILE_FILTER_RESULT = "geneProteinChangeACKY03.tsv"
PATH_FILE_NOFILTER_RESULT = "geneProteinChangeOther03.tsv"

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


def rest_get_variants_for_region_phenotype(chrom, start, end, gene, phenotype, num_top=20, pValueCutoff=0.0001, list_annotation=['missense_variant'], log=False):
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
                if row.get('pValue') < pValueCutoff:
                    map_variant = {'phenotype_code': phenotype, 'gene': gene, 'variant': row.get('varId'), 'p_value': row.get('pValue'), 'annotation': row.get('consequence')}

                    str_protein = get_protein_change(variant=map_variant.get('variant'))

                    # only include results with proteins
                    if str_protein:
                        map_variant['protein_change'] = str_protein

                        list_result.append(map_variant)



    # return
    return list_result



def rest_call_api_hgvs_notations(hgvs_notations, hg=38):
    # initialize
    result = None
    server = "https://rest.ensembl.org"
    if hg == 37:
        server = "https://grch37.rest.ensembl.org"
    ext = "/vep/human/hgvs"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    params = {"uniprot":'-', "refseq":'-', 'hgvs':'-'}
    data = {"hgvs_notations": hgvs_notations}

    # callt he server
    r = requests.post(server+ext, headers=headers, json=data, params=params)
    
    # if not r.ok:
    #     r.raise_for_status()
    #     sys.exit()
    if r.ok:
        result = r.json()
    
    return result


def get_protein_change(variant, log=False):
    '''
    returns the protein chgngee for a variant 
    '''
    # initialize 
    str_protein_change = None

    # change the variant to hgvs format ('17:g.7673537G>A')
    # input is '6:39033595:G:A'
    list_vars = variant.split(":")
    chrom = list_vars[0]
    pos = list_vars[1]
    ref = list_vars[2]
    alt = list_vars[3]
    str_hgvs = "{}:g.{}{}>{}".format(chrom, pos, ref, alt)

    # call the rest service
    json_response = rest_call_api_hgvs_notations(hgvs_notations=[str_hgvs], hg=37)

    # get the protein change
    if json_response[0].get('transcript_consequences'):
        for row in json_response[0].get('transcript_consequences'):
            if row.get('hgvsp') and not str_protein_change:
                temp_protein = row.get('hgvsp').split('.')[-1]
                str_protein_change = translate_protein_change(protein_change=temp_protein) 

    # return
    return str_protein_change


def translate_protein_change(protein_change):
    result = None

    # Dictionary to map three-letter amino acid codes to one-letter codes
    aa_three_to_one = {
        "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
        "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
        "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
        "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
        "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"
    }
    
    # Split the protein change notation
    old_aa = protein_change[:3]
    position = ''.join(filter(str.isdigit, protein_change))
    new_aa = protein_change[-3:]
    
    # Convert to single-letter notation
    if aa_three_to_one.get(old_aa) and aa_three_to_one.get(new_aa):
        old_aa_one = aa_three_to_one[old_aa]
        new_aa_one = aa_three_to_one[new_aa]
        result = f"p.{old_aa_one}{position}{new_aa_one}"

    # Return the translated notation
    return result

# # Example usage
# protein_change = "Gly168Ser"
# translated_change = translate_protein_change(protein_change)
# print(translated_change)


def is_this_a_lisine_change(protein_change, log=False):
    '''
    will return true/false 
    '''
    result = False

    if protein_change and protein_change[-1] in ['C', 'K', 'A', 'Y']:
        result = True

    return result


def write_tab_delimited_file(json_list, output_file, column_order=None):
    list_to_print = []

    # Check if the list is empty
    if not json_list:
        raise ValueError("The provided JSON list is empty")

    # If no column order is provided, use the keys from the first dictionary
    if column_order is None:
        column_order = json_list[0].keys()
    
    # build new list with only columns needed
    for item in json_list:
        map_temp = {}
        for col in column_order:
            map_temp[col] = item.get(col)
        list_to_print.append(map_temp)

    # Open the output file in write mode
    with open(output_file, 'w', newline='') as file:
        # Create a CSV writer object with tab delimiter
        writer = csv.DictWriter(file, fieldnames=column_order, delimiter='\t')

        # Write the header
        writer.writeheader()

        # Write the data
        # for item in json_list:
        for item in list_to_print:
            writer.writerow(item)

# main
if __name__ == "__main__":
    # initialize 
    gene = 'PPARG'
    gene = 'TP53'
    gene = 'KRAS'
    gene = 'GLP1R'

    list_gene = ['PAM', 'SLC30A8', 'MC4R', 'WIPI1', 'SOCS2', 'HNF1A', 'LRRTM3', 'GLP1R', 'ALDH2', 'CALR', 'DYNC2H1', 'TM6SF2', 'CDKN1B', 'ZNF76', 'PTPN23', 'CFAP221', 'IDH3G', 'SEZ6L', 'SSTR5', 'ZHX3']

    list3 = ['TPCN2', 'ASCC2', 'PAX4', 'PLXND1', 'TRIM51', 'MINDY1', 'ZFP91', 'MACF1', 'CPA1', 'POC5']
    list4 = ['PRIM1', 'ZAR1', 'SOS2', 'ACVR1C', 'PRRC2A', 'SETD9', 'PLCB3', 'UNC5C', 'RREB1', 'RLF']
    list5 = ['TP53', 'GCKR', 'OR4C46', 'PGM1', 'ZNF717', 'NYNRIN', 'APOE', 'ANGPTL4', 'KCNJ11', 'KIAA1755', 'GPNMB', 'PNPLA3', 'TSEN15', 'SPRED2', 'BDNF', 'CASR', 'CPNE4', 'MAFA', 'TSHZ3', 'TMEM175', 'RASGRP1', 'ZFHX3', 'HORMAD1', 'ABCB11', 'DHX58', 'DGAT1', 'LRFN2', 'TYRO3', 'FGFR1', 'NUCKS1', 'ING3', 'EP300', 'IGFBPL1', 'JARID2', 'SNX22', 'CNTD1', 'ABCB10', 'TH', 'JADE2', 'PTGFRN', 'DSTYK', 'INSR', 'IPO9', 'NEUROG3', 'FAIM2', 'ZNF641', 'LONRF1', 'EMILIN1', 'KDM5A', 'ZXDA', 'CREB3L2', 'KSR2', 'TFE3', 'YBX3', 'MFAP3', 'FAM13A', 'ENSG00000176349', 'LINC03012', 'PDE3A', 'TRIM37', 'TRIM40', 'TRPS1', 'ASCC1', 'BBS7', 'CCDC62']

    list_gene = list3 + list4
    list_gene = list5

    num = 20
    chrom = None
    start = None
    end = None
    list_variants = []
    list_filtered_variants = []
    list_nofiltered_variants = []

    # read in the phenotypes
    map_phenotypes = read_in_phenotypes()

    for index, gene in enumerate(list_gene):
        print("\n==== {}/{} processing gene: {}".format(index, len(list_gene), gene))
        # get the diseases for the gene
        list_gene_phenotypes, chrom, start, end = rest_get_disease_for_gene(gene=gene, num_top=num)

        # print
        for row in list_gene_phenotypes:
            print("for gene: {} (chrom: {}, start: {}, end: {}), got top {} disease: '{}' with pValue: {}".format(gene, 
                chrom, start, end, num, map_phenotypes.get(row.get('phenotype_code')), row.get('p_value')))

            # for each phenotype, find the variants
            list_temp = rest_get_variants_for_region_phenotype(chrom=chrom, start=start, end=end, gene=gene, phenotype=row.get('phenotype_code'), pValueCutoff=0.0005)
            print("found variants list of size: {}".format(len(list_temp)))

            # for each variant, add in the protein change
            for var in list_temp:
                var['phenotype'] = map_phenotypes.get(var.get('phenotype_code'))

            # add too the final list
            list_variants = list_variants + list_temp

    # filter the list based on amino acid substitution
    for var in list_variants:
        if var.get('protein_change') and is_this_a_lisine_change(var.get('protein_change')):
            list_filtered_variants.append(var)
        else:
            list_nofiltered_variants.append(var)

    # write out the files
    path_file = PATH_FILE_FILTER_RESULT
    print("got filtered variant list of size: {}".format(len(list_filtered_variants)))
    if len(list_filtered_variants) > 0: 
        write_tab_delimited_file(json_list=list_filtered_variants, output_file=path_file, column_order=['protein_change', 'variant', 'gene', 'phenotype', 'p_value'])
    else:
        print("got NO filtered variant list of size: {}".format(len(list_filtered_variants)))

    path_file = PATH_FILE_NOFILTER_RESULT
    print("got nofiltered variant list of size: {}".format(len(list_nofiltered_variants)))
    if len(list_nofiltered_variants) > 0: 
        write_tab_delimited_file(json_list=list_nofiltered_variants, output_file=path_file, column_order=['protein_change', 'variant', 'gene', 'phenotype', 'p_value'])
    else:
        print("NO FILE nofiltered variant list of size: {}".format(len(list_nofiltered_variants)))

    # print final list    
    print("got filter list of variants:\n{}".format(json.dumps(list_filtered_variants, indent=2)))




