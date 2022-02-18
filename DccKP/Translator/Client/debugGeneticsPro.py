
# imports
import logging
import requests
import sys
import json
import time


# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl

# list_subject = ["MESH:D056487", "MONDO:0005359", "SNOMEDCT:197358007", "MESH:D056487", "NCIT:C26991"]
list_subject = ["MONDO:0005359"]
list_categories_disease_or_phenotype = ["biolink:DiseaseOrPhenotypicFeature"]
list_categories_disease = ["biolink:Disease"]
list_categories_gene = ["biolink:Gene"]
list_subclass_predicates = ["biolink:subclass_of"]
list_predicates = ["biolink:has_real_world_evidence_of_association_with"]
url_arax = "https://arax.ncats.io/api/arax/v1.2/query"
url_aragorn = "https://aragorn.renci.org/1.1/query?answer_coalesce_type=all"
url_genepro = "https://translator.broadinstitute.org/genetics_provider/trapi/v1.2/query"
url_genepro = "http://0.0.0.0:7003//genetics_provider/trapi/v1.2/query"
url_node_descendants = "https://stars-app.renci.org/sparql-kp/query"
list_number_test = [10, 50, 100, 200, 500, 1000, 2000, 3000, 5000]
# list_number_test = [10000]

# methods

# main
if __name__ == "__main__":
    # initialize
    mondo_root = "MONDO:0000001"
    
    # get the subclasses of the main mondo node
    # list_nodes = tl.get_nodes_one_hop(url_node_descendants, None, [mondo_root], list_categories_disease_or_phenotype, None, list_subclass_predicates, log=True)

    # # build a list of curies for diseases
    # list_disease_inputs = []
    # for (id, name) in list_nodes:
    #     # logger.info("got result disease: {} - {}".format(id, name))
    #     list_disease_inputs.append(id)
    # logger.info("got subclass disease list of parent {} of count: {}".format(mondo_root, len(list_disease_inputs)))

    # # get the subclasses of the main mondo node
    # list_nodes = tl.get_nodes_one_hop(url_node_descendants, None, [mondo_root], list_categories_disease_or_phenotype, None, list_subclass_predicates, log=True)

    # # build a list of curies for diseases
    # list_two = []
    # for (id, name) in list_nodes:
    #     # logger.info("got result disease: {} - {}".format(id, name))
    #     list_two.append(id)
    # logger.info("got subclass disease list of parent {} of count: {}".format(mondo_root, len(list_two)))

    # list_diff = list(set(list_disease_inputs) - set(list_two))
    # logger.info("got {} elements in first list not in second".format(len(list_diff)))
    # list_diff = list(set(list_two) - set(list_disease_inputs))
    # logger.info("got {} elements in first list not in second".format(len(list_diff)))

    # sort the list descending
    # list_disease_inputs = sorted(list_disease_inputs, reverse=True)

    # test the service
    number_curies = 20
    # list_curie_input = list_disease_inputs[:number_curies]
    list_curie_input = ["MONDO:0100321",
                        "MONDO:0100086"]
    logger.info("testing genetics KP disease/gene links with disease list of size: {}".format(len(list_curie_input)))
    start = time.time()
    gene_nodes = tl.get_nodes_one_hop(url_genepro, list_curie_input, None, list_categories_disease, list_categories_gene, None, log=False)
    end = time.time()
    time_elapsed = end - start
    logger.info("got result gene list of size: {} in elapsed time (in seconds) of: {}".format(len(gene_nodes), time_elapsed))
    logger.info("")

    # list_curie_input2 = list_disease_inputs[:number_curies]
                # "MONDO:0100321": {
                #     "name": "viral disease or post-viral disorder"
                # "MONDO:0100086": {
                #     "name": "perinatal disease"
                # "MONDO:0045028": {
                #     "name": "radiation or chemically induced disorder"
                # "MONDO:0045024": {
                #     "name": "cell proliferation disorder"


    list_curie_input2 = ["MONDO:0100321",
                        "MONDO:0100086",
                        "MONDO:0045028",
                        "MONDO:0045024"]
    logger.info("testing genetics KP disease/gene links with disease list of size: {}".format(len(list_curie_input2)))
    start = time.time()
    gene_nodes2 = tl.get_nodes_one_hop(url_genepro, list_curie_input2, None, list_categories_disease, list_categories_gene, None, log=True)
    end = time.time()
    time_elapsed = end - start
    logger.info("got result gene list of size: {} in elapsed time (in seconds) of: {}".format(len(gene_nodes2), time_elapsed))
    logger.info("")


    list_diff = list(set(list_curie_input) - set(list_curie_input2))
    logger.info("got {} - {} elements diff in mondo curies".format(len(list(set(list_curie_input) - set(list_curie_input2))), len(list(set(list_curie_input2) - set(list_curie_input)))))
    list_diff = list(set(gene_nodes) - set(gene_nodes2))
    logger.info("got {}  - {} elements in first list not in second".format(len(list(set(gene_nodes) - set(gene_nodes2))), len(list(set(gene_nodes2) - set(gene_nodes)))))





