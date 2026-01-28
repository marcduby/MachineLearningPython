

# imports
import requests
import json

# constants
URL_PHENOTYPES = "https://bioindex-dev.hugeamp.org/api/bio/query/pigean-phenotypes?q=1"

# methods
def get_phenotype_map(endpoint_url: str=URL_PHENOTYPES) -> dict:
    """
    Sends a POST request to the given REST endpoint and returns
    a dict mapping phenotype -> phenotype_name.

    Args:
        endpoint_url (str): The API endpoint URL.
        payload (dict): The POST data (e.g. parameters, filters).

    Returns:
        dict: { phenotype: phenotype_name, ... }
    """
    try:
        response = requests.get(endpoint_url)
        response.raise_for_status()
        data = response.json()

        if "data" not in data:
            raise ValueError("Unexpected response format: missing 'data' field")

        phenotype_map = {
            item["phenotype"]: item["phenotype_name"]
            for item in data["data"]
            if "phenotype" in item and "phenotype_name" in item
        }

        return phenotype_map

    except requests.RequestException as e:
        print(f"HTTP error: {e}")
        return {}
    except Exception as e:
        print(f"Error: {e}")
        return {}



# main
if __name__ == "__main__":
    endpoint = "https://example.com/api/phenotypes"

    phenotype_map = get_phenotype_map()

    print(json.dumps(phenotype_map, indent=2))
    # for key, name in phenotype_map.items():
    #     print(f"{key}: {name}")
