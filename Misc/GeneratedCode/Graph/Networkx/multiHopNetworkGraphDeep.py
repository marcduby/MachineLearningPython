

import networkx as nx
# from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
# from langchain.graphs import NetworkxEntityGraph
from langchain.chains import GraphQAChain
from langchain_community.llms import OpenAI
# from langchain.llms import OpenAI
import json


import inspect
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

print(inspect.signature(NetworkxEntityGraph.__init__))


def create_detailed_graph():
    """Create a more detailed knowledge graph to demonstrate NetworkxEntityGraph features"""
    # graph = nx.Graph()
    # NOTE - need DiGraph for langchiain class used
    graph = nx.DiGraph()
    
    # Add nodes with properties
    nodes = {
        "Alice": {"entity_type": "person", "age": 28, "occupation": "engineer"},
        "Bob": {"entity_type": "person", "age": 34, "occupation": "manager"},
        "TechCorp": {"entity_type": "company", "industry": "technology", "size": "large"},
        "Project X": {"entity_type": "project", "status": "active", "priority": "high"}
    }
    
    # Add nodes with their properties
    for node, props in nodes.items():
        graph.add_node(node, **props)
    
    # Add edges with properties
    # edges = [
    #     ("Alice", "Bob", {"relationship": "reports_to", "duration": "2 years"}),
    #     ("Alice", "Project X", {"relationship": "works_on", "role": "lead developer"}),
    #     ("Bob", "TechCorp", {"relationship": "employed_by", "department": "Engineering"}),
    #     ("Project X", "TechCorp", {"relationship": "owned_by", "budget": "1M"})
    # ]
    edges = [
        # {"relationship": "reports_to", "source": "Alice", "target": "Bob", "properties": {"duration": "2 years"}},
        {"relationship": "reports_to", "source": "Alice", "target": "Bob", "properties": {"duration": "2 years"}},
        {"relationship": "works_on", "source": "Alice", "target": "Project X", "properties": {"role": "lead developer"}},
        {"relationship": "employs", "source": "TechCorp", "target": "Bob", "properties": {"department": "Engineering"}},
        {"relationship": "owns", "source": "TechCorp", "target": "Project X", "properties": {"budget": "1M"}}
    ]    

    # graph.add_edges_from(edges)
    for edge in edges:
        graph.add_edge(edge['source'], edge['target'], relation=edge['relationship'], properties=edge['properties'])

    # log
    print("got network graph: {}".format(json.dumps(nx.node_link_data(graph), indent=2)))
    # print("got network graph: {}".format(json.dumps(graph, indent=2)))
    return graph

def demonstrate_entity_graph_features(G):
    """Demonstrate key features of NetworkxEntityGraph"""
    
    # Create EntityGraph with custom property mappings
    try:
        entity_graph = NetworkxEntityGraph(
            graph=G
            # ,
            # source_key="source",
            # target_key="target",
            # edge_type_key="relationship",
            # # Optional: Specify property mappings
            # node_type_key="entity_type",
            # properties_key="properties"
        )
    except ValueError as e:
        print("\n\n -----------------------------------Initialization failed:", e)
    
    # Initialize GraphQAChain for querying
    chain = GraphQAChain.from_llm(
        llm=OpenAI(temperature=0),
        graph=entity_graph,
        verbose=True
    )
    
    # Demonstrate different types of queries
    queries = [
        "What is Alice's role in Project X?",
        "Who works at TechCorp and what are their roles?",
        "What is the connection between Alice and Bob, including all data.",
        "Describe the relationship between Alice and Bob, including duration.",
        "What are all the properties of Project X?",
        "How is TechCorp connected to Project X?"
    ]
    
    print("\nDemonstrating EntityGraph Query Capabilities:")
    print("=" * 50)
    for query in queries:
        print(f"\nQuery: {query}")

        # DEPRECATED - 
        # response = chain.run(query)
        response = chain.invoke(query)
        print(f"Response: {response}")
        print("-" * 50)
    
    return entity_graph, chain

def explore_graph_structure(entity_graph):
    """Explore and print the internal structure of the EntityGraph"""
    
    print("\nEntityGraph Internal Structure:")
    print("=" * 50)
    
    # Show node properties
    print("\nNode Properties:")
    for node in entity_graph.nodes:
        print(f"\nNode: {node}")
        print("Properties:", entity_graph.get_node_properties(node))
    
    # Show edge properties
    print("\nEdge Properties:")
    edges = entity_graph.get_edges()
    for edge in edges:
        print(f"\nEdge: {edge[0]} -> {edge[1]}")
        print("Properties:", entity_graph.get_edge_properties(edge[0], edge[1]))

def main():
    # Create and populate the graph
    G = create_detailed_graph()
    
    # Demonstrate features
    entity_graph, chain = demonstrate_entity_graph_features(G)
    
    # Explore structure
    # BUG - error
    # explore_graph_structure(entity_graph)
    
    # Example of a complex multi-hop query
    complex_query = """
    Starting from Alice, find all entities connected within 1 hop
    and describe their relationships and properties.
    """
    print("\nComplex Multi-hop Query:")
    print("=" * 50)
    print(f"Query: {complex_query}")
    # response = chain.run(complex_query)
    # print(f"Response: {response}")

    response = chain.invoke(complex_query)
    print(f"Response: {response.get('result')}")


if __name__ == "__main__":
    main()

    