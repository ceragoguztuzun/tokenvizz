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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def fasta_to_string(file_path):
    """
    Reads a .fa (FASTA) file and converts the sequence into a string.
    """
    with open(file_path, "r") as fasta_file:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        for record in tqdm(records, desc="Processing FASTA file"):
            sequence_str = str(record.seq)
            return sequence_str  # Return after processing the first record
    return ""

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

def tokenize_and_map(tokenizer, ref_dna, device, chunk_size=512):
    """
    Tokenizes the DNA sequence in chunks to prevent memory overload.
    """
    input_ids = []
    token_positions = []
    for i in range(0, len(ref_dna), chunk_size):
        chunk = ref_dna[i:i+chunk_size]
        tokenized_chunk = tokenizer(chunk, return_tensors='pt', return_offsets_mapping=True)
        input_ids.append(tokenized_chunk["input_ids"].to(device))
        offset_mapping = tokenized_chunk["offset_mapping"][0]
        tokens = tokenizer.convert_ids_to_tokens(tokenized_chunk["input_ids"][0].cpu())
        token_positions.extend([(token, start.item(), end.item()) for token, (start, end) in zip(tokens, offset_mapping) if token not in ["[CLS]", "[SEP]"]])

    return torch.cat(input_ids, dim=1), token_positions

def create_node_info(token_positions):
    return {i: {'string': token, 'position': f"{start}-{end}"} for i, (token, start, end) in enumerate(token_positions)}

def run_model(model, input_ids, layer_num=-1):
    """
    Runs the model on the input IDs and retrieves attention weights for a specific layer or the last layer.
    """
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

    # Add nodes to the graph based on token positions
    for i, (token, start, _) in enumerate(token_positions):
        G.add_node(i, label=token, start_position=start)
    
    initial_node_count = G.number_of_nodes()

    # Ensure the attention_weights are on the correct device (if they are not already)
    avg_attn_matrix = torch.mean(torch.cat([layer for layer in attention_weights], dim=1), dim=1)[0]
    
    # Move avg_attn_matrix to the correct device (optional, if needed)
    avg_attn_matrix = avg_attn_matrix.to(attention_weights[0].device)

    # Compute edge indices for the upper triangular part of the attention matrix
    edge_indices = torch.triu_indices(avg_attn_matrix.size(0), avg_attn_matrix.size(1), offset=1).to(attention_weights[0].device)
    
    # Extract edge weights from the attention matrix using the edge indices
    edge_weights = avg_attn_matrix[edge_indices[0], edge_indices[1]].to(attention_weights[0].device)
    
    # Mask out edges that have weights below the threshold
    mask = edge_weights >= threshold
    valid_edges = edge_indices[:, mask]
    valid_weights = edge_weights[mask]

    # Add edges to the graph, ensuring that nodes are within the initial node count
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

class DNADataset(Dataset):
    def __init__(self, ref_dna, chunk_size=512):
        self.ref_dna = ref_dna
        self.chunk_size = chunk_size

    def __len__(self):
        return (len(self.ref_dna) + self.chunk_size - 1) // self.chunk_size

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        return self.ref_dna[start:end]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process_on_gpu(rank, world_size, model_name, ref_dna):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    print(f"Processing on GPU {rank}")
    
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    model = DDP(model, device_ids=[rank])
    
    dataset = DNADataset(ref_dna)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    
    G = nx.Graph()
    node_info = {}
    
    for chunk in tqdm(dataloader, desc=f"GPU {rank} - Processing chunks", position=rank):
        input_ids, token_positions = tokenize_and_map(tokenizer, chunk[0], device)

        # Populate node_info with the desired format
        for i, (token, start, end) in enumerate(token_positions):
            node_info[i] = {
                "string": token,  # The DNA string (token)
                "position": f"{start}-{end}"  # Format the position as "start-end"
            }

        # Run the model and process attention weights
        try:
            attention_weights, _ = run_model(model.module, input_ids, layer_num=-1)
            chunk_G = create_graph(token_positions, attention_weights)
            G = nx.compose(G, chunk_G)
        
        except RuntimeError as e:
            print(f"GPU {rank} - RuntimeError occurred: {e}")
        except TypeError as e:
            print(f"GPU {rank} - TypeError occurred: {e}")

    # Save results for this rank, including the populated node_info
    torch.save((G, node_info), f'results_{rank}.pt')

    cleanup()

def main():
    print_system_info()
    clean_gpu()

    model_name = "jaandoui/DNABERT2-AttentionExtracted"

    if os.path.exists("ref_dna.txt"):
        with open("ref_dna.txt", "r") as f:
            ref_dna = f.read()
    else:
        ref_dna = fasta_to_string("/home/cxo147/ceRAG_viz/data/hg38.fa")
        #ref_dna = 'ATGCGCGTGAG'
        with open("ref_dna.txt", "w") as f:
            f.write(ref_dna)
    #'ATGCGATCGCTAGCTCGCGATCGATGATCGGAAGCTCTCTAGAGAGCTAGCTACCGCTAGCTACGACTAGCATCAGCTACGACTAG' #####
    world_size = torch.cuda.device_count()
    mp.spawn(process_on_gpu, args=(world_size, model_name, ref_dna), nprocs=world_size, join=True)

    combined_G = nx.Graph()
    combined_node_info = {}

    for rank in range(world_size):
        G, node_info = torch.load(f'results_{rank}.pt')
        print(f"Rank {rank} - node_info: {node_info}")  # Debugging to check content
        combined_G = nx.compose(combined_G, G)
        combined_node_info.update(node_info)


    for rank in range(world_size):
        G, node_info = torch.load(f'results_{rank}.pt')
        combined_G = nx.compose(combined_G, G)
        combined_node_info.update(node_info)

    print_graph_statistics(combined_G)

    with open('node_info.json', 'w') as f:
        json.dump(combined_node_info, f)
    print("Combined node information saved to node_info.json")

    saved_G_fn = "attention_graph_nothreshold"
    nx.write_edgelist(combined_G, f"{saved_G_fn}.csv", delimiter=",", data=['weight'])
    print(f"Combined graph saved to {saved_G_fn}.csv")

    print("Script completed successfully.")

if __name__ == "__main__":
    main()
