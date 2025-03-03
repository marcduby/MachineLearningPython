

from langchain_community.graphs import NetworkxEntityGraph
import networkx as nx

def convert_digraph_to_entity_graph(digraph: nx.DiGraph) -> NetworkxEntityGraph:
    """
    Convert a NetworkX DiGraph to a NetworkxEntityGraph.
    
    Args:
        digraph (nx.DiGraph): Input NetworkX directed graph
        
    Returns:
        NetworkxEntityGraph: Converted graph with entity relationships
    """
    # Create a new NetworkxEntityGraph
    entity_graph = NetworkxEntityGraph()
    
    # # Add all nodes with their attributes
    # for node, attrs in digraph.nodes(data=True):
    #     entity_graph.add_node(node, **attrs)
    
    # # Add all edges with their attributes
    # for source, target, attrs in digraph.edges(data=True):
    #     # NetworkxEntityGraph expects a 'type' attribute for relationships
    #     # If not present, use a default relationship type
    #     if 'type' not in attrs:
    #         attrs['type'] = 'related_to'
    #     entity_graph.add_edge(source, target, **attrs)

    # Add all nodes with their attributes
    for node, attrs in digraph.nodes(data=True):
        # NetworkxEntityGraph expects properties in a specific format
        properties = attrs.copy()
        entity_type = properties.pop('entity_type', 'entity')  # Default type if not specified
        # entity_graph.add_node(node, properties=properties, type=entity_type)
        entity_graph.add_node(node, type=entity_type)
    
    # Add all edges with their attributes
    for source, target, attrs in digraph.edges(data=True):
        # NetworkxEntityGraph expects properties in a specific format
        properties = attrs.copy()
        relation_type = properties.pop('type', 'related_to')  # Default type if not specified
        entity_graph.add_edge(source, target, relation=relation_type, properties=properties)

    return entity_graph

# Example usage:
if __name__ == "__main__":
    # Create a sample DiGraph
    G = nx.DiGraph()
    G.add_node("Person1", entity_type="person")
    G.add_node("Person2", entity_type="person")
    G.add_edge("Person1", "Person2", type="knows", weight=1)
    
    # Convert to NetworkxEntityGraph
    entity_G = convert_digraph_to_entity_graph(G)
    print(entity_G)

