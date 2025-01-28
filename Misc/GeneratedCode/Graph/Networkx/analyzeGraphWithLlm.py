

import networkx as nx
import openai

def query_graph(llm_response, G):
    # Example: LLM response might say "Find the shortest path between node A and node B"
    nodes = extract_nodes(llm_response)
    path = nx.shortest_path(G, source=nodes[0], target=nodes[1])
    return path

def extract_nodes(text):
    # Dummy function to extract node names from text
    return ["NodeA", "NodeB"]

# Setup your graph
G = nx.Graph()
G.add_edge("NodeA", "NodeB")
G.add_edge("NodeB", "NodeC")

# Example query from user
user_query = "What is the shortest path from Node A to Node B?"
response = openai.Completion.create(engine="davinci", prompt=user_query, max_tokens=50)

# Process the query against the graph
graph_result = query_graph(response['choices'][0]['text'], G)

# Use LLM to interpret the result back into natural language
final_response = f"The shortest path is: {' -> '.join(graph_result)}"
print(final_response)


