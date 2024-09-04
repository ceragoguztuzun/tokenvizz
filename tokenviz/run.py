import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import gc
import networkx as nx
import json
from Bio import SeqIO

def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def fasta_to_string(file_path):
    """
    Reads a .fa (FASTA) file and converts the sequence into a string.
    
    Parameters:
    file_path (str): The path to the .fa file.

    Returns:
    str: The DNA or protein sequence as a string.
    """
    # Open the FASTA file and parse the sequence
    with open(file_path, "r") as fasta_file:
        # SeqIO.parse returns an iterator; we can get the first (and usually only) record
        record = next(SeqIO.parse(fasta_file, "fasta"))
        
        # Convert the sequence object to a string
        sequence_str = str(record.seq)
    
    return sequence_str


def print_system_info():
    print(f'====================================')
    print(f'torch.__version__: {torch.__version__}')
    print(f'torch.version.cuda: {torch.version.cuda}')
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    print(f'transformers.__version__: {transformers.__version__}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f'====================================\n')
    return device

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name, return_dict=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    return model, tokenizer

def tokenize_and_map(tokenizer, ref_dna, device):
    tokenized = tokenizer(ref_dna, return_tensors='pt', return_offsets_mapping=True)
    input_ids = tokenized["input_ids"].to(device)
    offset_mapping = tokenized["offset_mapping"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    token_positions = [(token, start, end) for token, (start, end) in zip(tokens, offset_mapping) if token not in ["[CLS]", "[SEP]"]]
    return input_ids, token_positions

def create_node_info(token_positions):
    return {i: {'string': token, 'position': f"{start}-{end}"} for i, (token, start, end) in enumerate(token_positions)}

def run_model(model, input_ids):
    outputs = model(input_ids=input_ids, return_dict=True, output_attentions=True)
    if isinstance(outputs, tuple):
        attention_weights, last_hidden_states = outputs[-1], outputs[-2]
    else:
        attention_weights, last_hidden_states = outputs.attentions, outputs.last_hidden_state
    return attention_weights, last_hidden_states

def calculate_attention_bounds(attention_weights):
    min_weight, max_weight = float('inf'), float('-inf')
    for layer in attention_weights:
        for head in range(layer.size(1)):
            attn_matrix = layer[0, head].detach().cpu().numpy()
            min_weight = min(min_weight, attn_matrix.min())
            max_weight = max(max_weight, attn_matrix.max())
    return min_weight, max_weight

def create_graph(token_positions, attention_weights, threshold=0):
    G = nx.Graph()
    for i, (token, start, _) in enumerate(token_positions):
        G.add_node(i, label=token, start_position=start)
    
    initial_node_count = G.number_of_nodes()
    
    for layer in attention_weights:
        avg_attn_matrix = torch.mean(layer[0], dim=0).detach().cpu().numpy()
        for i in range(avg_attn_matrix.shape[0]):
            for j in range(i + 1, avg_attn_matrix.shape[1]):
                if i < initial_node_count and j < initial_node_count:
                    weight = avg_attn_matrix[i, j]
                    if weight >= threshold:
                        if G.has_edge(i, j):
                            G[i][j]['weight'] = max(G[i][j]['weight'], weight)
                        else:
                            G.add_edge(i, j, weight=weight)
    
    assert G.number_of_nodes() == initial_node_count
    return G

def print_graph_statistics(G):
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
#################################################################################################
def main():
    device = print_system_info()
    clean_gpu()

    model_name = "jaandoui/DNABERT2-AttentionExtracted"
    model, tokenizer = load_model_and_tokenizer(model_name)
    model = model.to(device)

    #ref_dna = "AGCTTAGCTAGCTAGCTGACT"
    ref_dna = fasta_to_string("/home/cxo147/ceRAG_viz/data/hg38.fa")
    with open("ref_dna.txt", "w") as f:
        f.write(ref_dna)
    input_ids, token_positions = tokenize_and_map(tokenizer, ref_dna, device)
    
    node_info = create_node_info(token_positions)
    print("\nNode Information:")
    print(node_info)

    # Save node_info to JSON file
    with open('node_info.json', 'w') as f:
        json.dump(node_info, f)
    print("Node information saved to node_info.json")

    try:
        attention_weights, _ = run_model(model, input_ids)
        min_weight, max_weight = calculate_attention_bounds(attention_weights)
        print(f"Min attention weight: {min_weight}")
        print(f"Max attention weight: {max_weight}")

        G = create_graph(token_positions, attention_weights)
        print_graph_statistics(G)

        print("\nNode names (tokens):")
        for i, (token, _, _) in enumerate(token_positions):
            print(f"Node {i}: {token}")

        saved_G_fn = "attention_graph_nothreshold"
        nx.write_edgelist(G, f"{saved_G_fn}.csv", delimiter=",", data=['weight'])
        print(f"Graph saved to {saved_G_fn}.csv")

    except RuntimeError as e:
        print(f"RuntimeError occurred: {e}")
    except TypeError as e:
        print(f"TypeError occurred: {e}")

    print("Script completed successfully.")

if __name__ == "__main__":
    main()
