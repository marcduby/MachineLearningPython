
# imports
import logging
import requests
import sys
import json


# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl

# list_subject = ["MESH:D056487", "MONDO:0005359", "SNOMEDCT:197358007", "MESH:D056487", "NCIT:C26991"]
list_subject = ["MONDO:0005359"]
list_categories = ["biolink:DiseaseOrPhenotypicFeature"]
list_predicates = ["biolink:has_real_world_evidence_of_association_with"]

url_icees = "https://icees.renci.org:16341/query?reasoner=true&verbose=false"
url_cohd = "https://trapi-dev.cohd.io/api/query"
url_arax = "https://arax.ncats.io/api/arax/v1.2/query"
url_aragorn = "https://aragorn.renci.org/1.1/query?answer_coalesce_type=all"
url_genepro = "https://translator.broadinstitute.org/genetics_provider/trapi/v1.2/query"

# methods


# main
if __name__ == "__main__":
    # get the list of nodes from the first step
    # list_nodes = tl.get_nodes_one_hop(url_arax, list_subject, None, list_categories, list_categories, list_predicates, log=True)
    list_nodes = tl.get_nodes_one_hop(url_icees, list_subject, None, list_categories, list_categories, list_predicates, log=True)

    # build a list of curies for diseases
    list_disease_inputs = []
    for (id, name) in list_nodes:
        logger.info("got result disease: {} - {}".format(id, name))
        list_disease_inputs.append(id)

    # print the names of the diseases


    # query the genetics provider
    # list_genepro_nodes = tl.get_nodes_one_hop(url_genepro, list_disease_inputs, None, list_categories, ["biolink:Gene"], None, log=True)
