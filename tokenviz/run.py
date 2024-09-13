import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import gc
import networkx as nx
import json
from Bio import SeqIO
import os
from tqdm import tqdm
import torch.multiprocessing as mp

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
        records = list(SeqIO.parse(fasta_file, "fasta"))
        
        # Use tqdm to show progress
        for record in tqdm(records, desc="Processing FASTA file"):
            # Convert the sequence object to a string
            sequence_str = str(record.seq)
            return sequence_str  # Return after processing the first record

    return ""  # Return empty string if no records found


def print_system_info():
    print(f'====================================')
    print(f'torch.__version__: {torch.__version__}')
    print(f'torch.version.cuda: {torch.version.cuda}')
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    print(f'transformers.__version__: {transformers.__version__}')
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f'====================================\n')

def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name, return_dict=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)
    model = model.to(device)
    return model, tokenizer

def tokenize_and_map(tokenizer, ref_dna, device):
    tokenized = tokenizer(ref_dna, return_tensors='pt', return_offsets_mapping=True)
    input_ids = tokenized["input_ids"].to(device)
    offset_mapping = tokenized["offset_mapping"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    token_positions = [(token, start.item(), end.item()) for token, (start, end) in zip(tokens, offset_mapping) if token not in ["[CLS]", "[SEP]"]]
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
    min_weight = torch.min(torch.cat([layer.view(-1) for layer in attention_weights]))
    max_weight = torch.max(torch.cat([layer.view(-1) for layer in attention_weights]))
    return min_weight.item(), max_weight.item()

def create_graph(token_positions, attention_weights, threshold=0):
    G = nx.Graph()
    for i, (token, start, _) in enumerate(token_positions):
        G.add_node(i, label=token, start_position=start)
    
    initial_node_count = G.number_of_nodes()
    
    avg_attn_matrix = torch.mean(torch.cat([layer for layer in attention_weights], dim=1), dim=1)[0]
    edge_indices = torch.triu_indices(avg_attn_matrix.size(0), avg_attn_matrix.size(1), offset=1)
    edge_weights = avg_attn_matrix[edge_indices[0], edge_indices[1]]
    
    mask = edge_weights >= threshold
    valid_edges = edge_indices[:, mask]
    valid_weights = edge_weights[mask]
    
    for i, j, weight in zip(valid_edges[0], valid_edges[1], valid_weights):
        i, j = i.item(), j.item()
        if i < initial_node_count and j < initial_node_count:
            if G.has_edge(i, j):
                G[i][j]['weight'] = max(G[i][j]['weight'], weight.item())
            else:
                G.add_edge(i, j, weight=weight.item())
    
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

def process_on_gpu(gpu_id, model_name, ref_dna):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Processing on GPU {gpu_id}")
    
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    input_ids, token_positions = tokenize_and_map(tokenizer, ref_dna, device)
    
    node_info = create_node_info(token_positions)
    
    try:
        attention_weights, _ = run_model(model, input_ids)
        min_weight, max_weight = calculate_attention_bounds(attention_weights)
        print(f"GPU {gpu_id} - Min attention weight: {min_weight}")
        print(f"GPU {gpu_id} - Max attention weight: {max_weight}")

        G = create_graph(token_positions, attention_weights)
        print(f"GPU {gpu_id} - Graph created")
        
        return G, node_info
    
    except RuntimeError as e:
        print(f"GPU {gpu_id} - RuntimeError occurred: {e}")
    except TypeError as e:
        print(f"GPU {gpu_id} - TypeError occurred: {e}")

def main():
    print_system_info()
    clean_gpu()

    model_name = "jaandoui/DNABERT2-AttentionExtracted"

    if os.path.exists("ref_dna.txt"):
        with open("ref_dna.txt", "r") as f:
            ref_dna = f.read()
    else:
        ref_dna = fasta_to_string("/home/cxo147/ceRAG_viz/data/hg38.fa")
        with open("ref_dna.txt", "w") as f:
            f.write(ref_dna)

    ref_dna = 'ATGCGTGA'
    mp.set_start_method('spawn')
    
    clean_gpu()
    with mp.Pool(processes=3) as pool:
        results = pool.starmap(process_on_gpu, [(1, model_name, ref_dna), (2, model_name, ref_dna), (3, model_name, ref_dna)])

    # Combine results from all GPUs
    combined_G = nx.Graph()
    combined_node_info = {}
    
    if results is None:
        print("Error: Processing failed due to CUDA out of memory. No results were generated.")
        return  # or sys.exit(1) if you want to terminate the script

    for G, node_info in results:
        if G is not None:
            combined_G = nx.compose(combined_G, G)
            combined_node_info.update(node_info)

    print_graph_statistics(combined_G)

    # Save combined node_info to JSON file
    with open('node_info.json', 'w') as f:
        json.dump(combined_node_info, f)
    print("Combined node information saved to node_info.json")

    saved_G_fn = "attention_graph_nothreshold"
    nx.write_edgelist(combined_G, f"{saved_G_fn}.csv", delimiter=",", data=['weight'])
    print(f"Combined graph saved to {saved_G_fn}.csv")

    print("Script completed successfully.")

if __name__ == "__main__":
    main()
