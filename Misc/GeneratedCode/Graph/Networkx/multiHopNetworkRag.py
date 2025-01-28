

import networkx as nx
from langchain.chains import GraphQAChain
from langchain.graphs import NetworkxEntityGraph
from langchain.llms import OpenAI
import matplotlib.pyplot as plt

def create_sample_graph():
    """Create a sample knowledge graph for demonstration"""
    G = nx.Graph()
    
    # Add nodes
    people = ["Alice", "Bob", "Charlie", "David", "Eve"]
    for person in people:
        G.add_node(person, entity_type="person")
    
    # Add relationships
    relationships = [
        ("Alice", "Bob", "friend"),
        ("Bob", "Charlie", "colleague"),
        ("Charlie", "David", "mentor"),
        ("David", "Eve", "supervisor"),
        ("Alice", "Eve", "classmate")
    ]
    
    for source, target, relation in relationships:
        G.add_edge(source, target, relationship=relation)
    
    return G

def analyze_multi_hop_relationships(graph, start_node, max_hops=3):
    """
    Analyze relationships between nodes up to specified number of hops
    """
    # Convert NetworkX graph to LangChain EntityGraph
    entity_graph = NetworkxEntityGraph(
        graph,
        source_key="source",
        target_key="target",
        edge_type_key="relationship"
    )
    
    # Initialize GraphQAChain
    chain = GraphQAChain.from_llm(
        llm=OpenAI(temperature=0),
        graph=entity_graph,
        verbose=True
    )
    
    # Generate questions for different hop distances
    for hops in range(1, max_hops + 1):
        question = f"What are the {hops}-hop connections from {start_node}?"
        response = chain.run(question)
        print(f"\n{hops}-hop analysis:")
        print(response)
        
    return chain

def visualize_graph(graph):
    """
    Visualize the graph with node labels and edge relationships
    """
    pos = nx.spring_layout(graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                          node_size=500)
    nx.draw_networkx_labels(graph, pos)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(graph, 'relationship')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plt.title("Knowledge Graph Visualization")
    plt.axis('off')
    plt.show()

def main():
    # Create sample graph
    G = create_sample_graph()
    
    # Visualize the graph
    visualize_graph(G)
    
    # Analyze multi-hop relationships starting from 'Alice'
    chain = analyze_multi_hop_relationships(G, "Alice", max_hops=3)
    
    # Example of custom query
    custom_query = "What is the relationship path between Alice and David?"
    result = chain.run(custom_query)
    print("\nCustom query result:")
    print(result)

if __name__ == "__main__":
    main()


    