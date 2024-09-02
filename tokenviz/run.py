import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import gc
import networkx as nx

def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()



print(f'====================================\ntorch.__version__: {torch.__version__}')
print(f'torch.version.cuda: {torch.version.cuda}')
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'transformers.__version__: {transformers.__version__}')

# Set device to all available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
print(f"Number of GPUs available: {torch.cuda.device_count()}\n====================================\n")

clean_gpu()
# Load model and tokenizer
model_name = "jaandoui/DNABERT2-AttentionExtracted"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = BertConfig.from_pretrained(model_name, return_dict=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)

# Move the model to multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model = model.to(device)

# Minimal DNA sequence for testing
ref_dna = "AGCTTAGCTAGCTAGCTGACT"

# Tokenize inputs and track token positions
tokenized = tokenizer(ref_dna, return_tensors='pt', return_offsets_mapping=True)
input_ids = tokenized["input_ids"].to(device)
offset_mapping = tokenized["offset_mapping"][0].tolist()

# Create a mapping of tokens to their positions in the original sequence
tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
token_positions = []
for token, (start, end) in zip(tokens, offset_mapping):
    if token not in ["[CLS]", "[SEP]"]:
        token_positions.append((token, start, end))

# Create a dictionary of node information
node_info = {}
for i, (token, start, end) in enumerate(token_positions):
    node_info[i] = {
        'string': token,
        'position': f"{start}-{end}"
    }

# Print the node information dictionary
print("\nNode Information:")
print(node_info)

# Run the model
try:
    outputs = model(input_ids=input_ids, return_dict=True, output_attentions=True)

    # If the output is still a tuple, fallback to manually extracting
    if isinstance(outputs, tuple):
        attention_weights = outputs[-1] 
        last_hidden_states = outputs[-2]
    else:
        attention_weights = outputs.attentions
        last_hidden_states = outputs.last_hidden_state

    # Calculate the minimum and maximum attention weights across all layers and heads
    min_weight = float('inf')
    max_weight = float('-inf')

    for layer in range(len(attention_weights)):
        for head in range(attention_weights[layer].size(1)):  # Iterate over heads
            attn_matrix = attention_weights[layer][0, head].detach().cpu().numpy()  # shape [sequence_length, sequence_length]
            min_weight = min(min_weight, attn_matrix.min())
            max_weight = max(max_weight, attn_matrix.max())

    print(f"Min attention weight: {min_weight}")
    print(f"Max attention weight: {max_weight}")

    # Define a threshold for the attention weights to determine edge existence
    threshold = 0#0.5 * (min_weight + max_weight)  # Example: median threshold

    # Create a graph using networkx
    G = nx.Graph()
    # Add nodes (tokens) to the graph with start position as an attribute
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
    print(f"Number of tokens...: {len(tokens)}")
    for i, (token, start, end) in enumerate(token_positions):
        G.add_node(i, label=token, start_position=start)
        print(f"Node added {i}: {token} (start position: {start})")

    initial_node_count = G.number_of_nodes()

    # Add edges between nodes based on the average attention weights across heads and threshold
    for layer in range(len(attention_weights)):
        # Average attention weights across all heads for this layer
        avg_attn_matrix = torch.mean(attention_weights[layer][0], dim=0).detach().cpu().numpy()
        for i in range(avg_attn_matrix.shape[0]):
            for j in range(i + 1, avg_attn_matrix.shape[1]):
                # Ensure that i and j are within the bounds of the existing nodes
                if i < initial_node_count and j < initial_node_count:
                    weight = avg_attn_matrix[i, j]
                    if weight >= threshold:
                        if G.has_edge(i, j):
                            # If edge exists, update weight
                            G[i][j]['weight'] = max(G[i][j]['weight'], weight)
                        else:
                            # If edge doesn't exist, add it
                            G.add_edge(i, j, weight=weight)
                else:
                    #print(f"Skipping edge between non-existent nodes: {i} -> {j}")
                    pass

    assert G.number_of_nodes() == initial_node_count


    # Generate and display graph statistics report
    print("\nGraph Statistics Report:")
    print("-------------------------")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Graph density: {nx.density(G):.4f}")
    print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
    
    if nx.is_connected(G):
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.4f}")
    else:
        print("Graph is not connected. Cannot compute average shortest path length.")
    
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    
    degree_sequence = [d for n, d in G.degree()]
    print(f"Average degree: {sum(degree_sequence) / len(degree_sequence):.4f}")
    print(f"Maximum degree: {max(degree_sequence)}")
    print(f"Minimum degree: {min(degree_sequence)}")
    print("-------------------------")

    print("\nNode names (tokens):")
    for i, token in enumerate(tokens):
        print(f"Node {i}: {token}")

    # Save the graph as csv with node attributes
    saved_G_fn = "attention_graph_nothreshold"
    nx.write_edgelist(G, f"{saved_G_fn}.csv", delimiter=",", data=['weight'])
    print(f"Graph saved to {saved_G_fn}.csv")

except RuntimeError as e:
    print(f"RuntimeError occurred: {e}")
except TypeError as e:
    print(f"TypeError occurred: {e}")

print("Script completed successfully.")
