import argparse  # Added for argument parsing
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import gc
import scipy.sparse as sp  # Importing SciPy for sparse matrices
import json
import pysam
import os
import re
import logging
import traceback
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

def clean_sequence(seq):
    seq = seq.upper()
    return re.sub(r'[^ATCG]', 'N', seq)

def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def print_system_info():
    logging.info('====================================')
    logging.info(f'torch.__version__: {torch.__version__}')
    logging.info(f'torch.version.cuda: {torch.version.cuda}')
    logging.info(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    logging.info(f'transformers.__version__: {transformers.__version__}')
    logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    logging.info('====================================\n')

class GPUFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rankname = f"GPU {rank}" if isinstance(rank, int) else str(rank)

    def filter(self, record):
        record.rankname = self.rankname
        return True

class CustomFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'rankname'):
            record.rankname = 'MAIN'  # Default value when rankname is missing
        return super().format(record)

def setup_logging(rank, log_dir=None):
    if isinstance(rank, int):
        log_filename = f'process_{rank}.log'
        rankname = f'GPU {rank}'
    else:
        log_filename = 'main.log'
        rankname = str(rank)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
    else:
        log_path = log_filename

    # Create a logger specific to the process
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set to INFO to reduce overhead

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)  # Set to INFO

    # Create a custom formatter using new-style formatting
    formatter = CustomFormatter(
        fmt='{asctime} [{rankname}] {levelname}: {message}',
        datefmt='%Y-%m-%d %H:%M:%S',
        style='{'
    )

    # Set the formatter for the handler
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Add the custom GPUFilter to include rankname in log records
    logger.addFilter(GPUFilter(rank))

def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name, return_dict=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)
    model = model.to(device)

    max_seq_length = model.config.max_position_embeddings
    logging.info(f"Model's maximum sequence length: {max_seq_length}")

    return model, tokenizer, max_seq_length

def tokenize_and_map(tokenizer, ref_dna_chunks, device, max_seq_length, offsets, kmer_size):
    tokenized = tokenizer(
        list(ref_dna_chunks),
        return_tensors='pt',
        truncation=True,
        max_length=max_seq_length,
        padding=True,
    )
    input_ids = tokenized["input_ids"].to(device)
    all_token_positions = []
    
    for i in range(len(ref_dna_chunks)):
        tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][i].cpu())
        offset = offsets[i]
        token_positions = []
        seq_idx = 0  # Position in the original sequence
        for token in tokens:
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            # Compute start and end positions based on seq_idx and kmer_size
            start = seq_idx + offset
            end = start + kmer_size
            token_positions.append((token, start, end))
            seq_idx += 1  # Move to the next position in the sequence

        logging.debug(f"Chunk {i}: {len(token_positions)} tokens processed.")
        all_token_positions.append(token_positions)
    logging.debug(f"Total token positions returned: {len(all_token_positions)}")
    return input_ids, all_token_positions

def run_model(model, input_ids, layer_num=-1):
    outputs = model(input_ids=input_ids, return_dict=True, output_attentions=True)
    if isinstance(outputs, tuple):
        attention_weights, last_hidden_state = outputs[-1], outputs[-2]
    else:
        attention_weights, last_hidden_state = outputs.attentions, outputs.last_hidden_state
    return attention_weights, last_hidden_state

def create_graph(batch_token_positions, batch_attn_weights, threshold=0.01):
    """
    Constructs edge lists, weights, and node information from token positions and attention weights.

    Args:
        batch_token_positions (List[List[Tuple]]): Token positions per sample.
        batch_attn_weights (List[List[Tensor]]): Attention weights per sample across layers.
        threshold (float): Minimum attention weight to consider creating an edge.

    Returns:
        Tuple[List[Tuple[int, int]], List[float], Dict[int, Dict[str, str]], Set[int]]:
            - List of edges represented as (node_i, node_j).
            - List of corresponding weights.
            - Dictionary mapping node_id to {'string': ..., 'position': ...}.
            - Set of unique node_ids.
    """
    edges = []
    weights = []
    node_info = {}  # mapping node_id -> {'string': ..., 'position': ...}
    unique_node_ids = set()

    for i, (token_positions, attn_weights) in enumerate(zip(batch_token_positions, batch_attn_weights)):
        node_ids = []
        for (token, start, end) in token_positions:
            if token == "[UNK]":
                continue  # Skip [UNK] tokens
            node_ids.append(start)
            unique_node_ids.add(int(start))  # Ensure start is a native Python int
            # Collect node info
            node_info[int(start)] = {
                'string': token,
                'position': f"{int(start)}-{int(end)}"
            }

        seq_len = len(node_ids)
        if seq_len == 0:
            logging.warning(f"Sample {i}: Sequence length is zero after excluding [UNK], skipping graph creation.")
            continue  # Skip this sample

        # Stack attention weights across layers: [num_layers, num_heads, seq_len, seq_len]
        try:
            attn_stack = torch.stack(attn_weights)  # [num_layers, num_heads, seq_len, seq_len]
        except Exception as e:
            logging.error(f"Sample {i}: Error stacking attention weights: {e}")
            logging.error(traceback.format_exc())
            continue  # Skip this sample

        # Average over layers and heads: [seq_len, seq_len]
        avg_attn_matrix = attn_stack.mean(dim=[0, 1])  # Shape: [seq_len, seq_len]
        logging.debug(f"Sample {i}: Averaged attention matrix shape: {avg_attn_matrix.shape}")

        # Find upper triangular indices to avoid duplicate edges
        edge_indices = torch.triu_indices(seq_len, seq_len, offset=1).to(avg_attn_matrix.device)
        edge_weights = avg_attn_matrix[edge_indices[0], edge_indices[1]]

        # Apply threshold to filter weak connections
        mask = edge_weights >= threshold
        valid_edges = edge_indices[:, mask]
        valid_weights = edge_weights[mask]

        logging.debug(f"Sample {i}: Number of valid edges after thresholding: {valid_edges.shape[1]}")

        if valid_edges.shape[1] == 0:
            logging.warning(f"Sample {i}: No valid edges after thresholding.")
            continue  # Skip this sample

        # Collect edges and weights
        for idx_i, idx_j, weight in zip(valid_edges[0], valid_edges[1], valid_weights):
            node_i = int(node_ids[idx_i.item()])  # Convert to native Python int
            node_j = int(node_ids[idx_j.item()])  # Convert to native Python int
            edges.append((node_i, node_j))
            weights.append(float(weight.item()))  # Ensure weight is a native Python float

    # Return edges, weights, node_info, and unique node_ids
    return edges, weights, node_info, unique_node_ids

def print_graph_statistics(total_edges, total_nodes):
    logging.info("\nGraph Statistics Report:")
    logging.info("-------------------------")
    logging.info(f"Number of nodes: {len(total_nodes)}")
    logging.info(f"Number of edges: {len(total_edges)}")
    logging.info("-------------------------")

class DNADataset(Dataset):
    def __init__(self, file_path, rank, world_size, chunk_size, max_chunks=None):
        """
        Initialize the DNADataset.

        Args:
            file_path (str): Path to the FASTA file.
            rank (int): Rank of the current process.
            world_size (int): Total number of processes.
            chunk_size (int): Size of each chunk to process.
            max_chunks (int, optional): Maximum number of chunks to process for testing.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size
        self.fasta = pysam.FastaFile(self.file_path)
        self.references = self.fasta.references
        self.lengths = self.fasta.lengths
        self.chunk_indices = self._compute_chunk_indices(max_chunks)

    def _compute_chunk_indices(self, max_chunks):
        total_chunks = []
        for ref, length in zip(self.references, self.lengths):
            num_chunks = (length + self.chunk_size - 1) // self.chunk_size
            indices = [(ref, i * self.chunk_size) for i in range(num_chunks)]
            total_chunks.extend(indices)
        # Distribute indices among processes
        distributed_chunks = total_chunks[self.rank::self.world_size]
        logging.info(f"GPU {self.rank}: Assigned {len(distributed_chunks)} chunks.")

        # If max_chunks is set, limit the number of chunks
        if max_chunks is not None:
            distributed_chunks = distributed_chunks[:max_chunks]
            logging.info(f"GPU {self.rank}: Limited to first {max_chunks} chunks for testing.")

        return distributed_chunks

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        ref_name, start = self.chunk_indices[idx]
        end = start + self.chunk_size
        seq_length = self.fasta.get_reference_length(ref_name)
        if end > seq_length:
            end = seq_length  # Adjust end if it exceeds the sequence length
        sequence = self.fasta.fetch(ref_name, start, end)
        return sequence, start

    # Handle pickling by excluding non-picklable attributes
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['fasta']
        del state['references']
        del state['lengths']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reopen the FastaFile in the worker process
        self.fasta = pysam.FastaFile(self.file_path)
        self.references = self.fasta.references
        self.lengths = self.fasta.lengths

def process_on_gpu(rank, world_size, model_name, available_gpus, dataset_path, kmer_size, threshold, max_chunks=None):
    setup_logging(rank)
    gpu_id = available_gpus[rank]
    device = torch.device(f"cuda:{gpu_id}")
    logging.info(f"Process {rank} is using GPU {gpu_id}")

    model, tokenizer, max_seq_length = load_model_and_tokenizer(model_name, device)
    chunk_size = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    logging.info(f"Setting chunk size to {chunk_size} based on model's max sequence length.")

    # Initialize the dataset with max_chunks for testing
    dataset = DNADataset(dataset_path, rank, world_size, chunk_size, max_chunks=max_chunks)

    # Adjust DataLoader for testing
    dataloader = DataLoader(
        dataset,
        batch_size=16,         # You can make this configurable if needed
        num_workers=4,         # You can make this configurable if needed
        drop_last=False        # Keep all data
    )

    all_edges = []
    all_weights = []
    all_unique_nodes = set()
    all_node_info = {}

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"GPU {rank} - Processing chunks", position=rank)):
        chunks, chunk_starts = batch
        chunks = [clean_sequence(chunk) for chunk in chunks]
        chunk_starts = chunk_starts.numpy()
        try:
            input_ids, all_token_positions = tokenize_and_map(
                tokenizer, chunks, device, max_seq_length, chunk_starts, kmer_size
            )

            logging.info(f"Batch {batch_idx}: Token positions batch size: {len(all_token_positions)}")

            # Run the model
            with torch.cuda.amp.autocast():
                attention_weights, _ = run_model(model, input_ids)

            logging.info(f"Batch {batch_idx}: Number of attention layers: {len(attention_weights)}")
            for layer_idx, layer_attn in enumerate(attention_weights):
                logging.debug(f"Batch {batch_idx}: Layer {layer_idx} attention shape: {layer_attn.shape}")

            # Reorganize attention_weights per sample
            batch_size = input_ids.size(0)
            attention_weights_per_sample = [[] for _ in range(batch_size)]

            for layer_idx, layer_attn in enumerate(attention_weights):
                if layer_attn.size(0) != batch_size:
                    logging.error(f"Batch {batch_idx}: Layer {layer_idx} has batch size {layer_attn.size(0)}, expected {batch_size}")
                    continue  # Skip this layer

                for i in range(batch_size):
                    attention_weights_per_sample[i].append(layer_attn[i])

            logging.info(f"Batch {batch_idx}: Attention weights per sample: {len(attention_weights_per_sample)} samples")

            # Ensure the batch size matches
            if len(all_token_positions) != len(attention_weights_per_sample):
                logging.error(f"Batch {batch_idx}: Mismatch between token positions and attention matrices: {len(all_token_positions)} vs {len(attention_weights_per_sample)}")
                continue  # Skip this batch

            # Process the batch
            edges, weights, node_info, unique_nodes = create_graph(all_token_positions, attention_weights_per_sample, threshold=threshold)
            all_edges.extend(edges)
            all_weights.extend(weights)
            all_unique_nodes.update(unique_nodes)
            all_node_info.update(node_info)

        except Exception as e:
            logging.error(f"GPU {rank} - Error occurred: {e}")
            logging.error(traceback.format_exc())
            continue  # Skip this batch

    # Save the edges, weights, and node_info to a file for later aggregation
    if all_edges:
        edge_file = f'results_{rank}.json'
        with open(edge_file, 'w') as f:
            json.dump({
                'edges': all_edges,                     # List of tuples with native Python ints
                'weights': all_weights,                 # List of native Python floats
                'unique_nodes': list(all_unique_nodes), # List of native Python ints
                'node_info': all_node_info              # Dict mapping node_id to {'string': ..., 'position': ...}
            }, f)
        logging.info(f"GPU {rank}: Saved {len(all_edges)} edges and node information to {edge_file}")
    else:
        logging.info(f"GPU {rank}: No edges to save.")

########################################################################3##########################################3#########################333############################
def main():
    parser = argparse.ArgumentParser(description="Process DNA sequences with DNABERT2 and extract attention graphs.")

    # Required arguments
    parser.add_argument('--kmer_size', type=int, default=6, help='Size of the k-mer (default: 6)')
    parser.add_argument('--saved_matrix_fn', type=str, default="attention_graph_nothreshold.npz", help='Filename to save the adjacency matrix (default: attention_graph_nothreshold.npz)')
    parser.add_argument('--node_info_fn', type=str, default="node_info.json", help='Filename to save node information (default: node_info.json)')
    parser.add_argument('--original_gpus', type=int, nargs='+', default=[5], help='List of GPU IDs to use (default: [5])')
    parser.add_argument('--model_name', type=str, default="jaandoui/DNABERT2-AttentionExtracted", help='Name of the pre-trained model (default: jaandoui/DNABERT2-AttentionExtracted)')
    parser.add_argument('--dataset_path', type=str, default="/home/cxo147/ceRAG_viz/data/hg38.fa", help='Path to the FASTA dataset (default: /home/cxo147/ceRAG_viz/data/hg38.fa)')

    # Optional arguments (suggestions)
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold for attention weights to create edges (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for DataLoader (default: 4)')
    parser.add_argument('--test_mode', action='store_true', help='Enable test mode with limited data processing')
    parser.add_argument('--max_chunks', type=int, default=10, help='Maximum number of chunks to process in test mode (default: 10)')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory to save log files (default: current directory)')

    args = parser.parse_args()

    # Extract arguments
    kmer_size = args.kmer_size
    saved_matrix_fn = args.saved_matrix_fn
    node_info_fn = args.node_info_fn
    original_gpus = args.original_gpus
    model_name = args.model_name
    dataset_path = args.dataset_path
    threshold = args.threshold
    batch_size = args.batch_size
    num_workers = args.num_workers
    test_mode = args.test_mode
    max_chunks = args.max_chunks if test_mode else None
    log_dir = args.log_dir

    # Pass the log_dir to the main logger
    setup_logging('MAIN', log_dir=log_dir)
    logging.info("Starting main process")
    print_system_info()
    clean_gpu()

    # Specify the GPUs you want to use
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, original_gpus))
    logging.info(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Adjust GPU indices after setting CUDA_VISIBLE_DEVICES
    available_gpus = list(range(len(original_gpus)))
    world_size = len(available_gpus)
    logging.info(f"Adjusted available GPUs: {available_gpus}")

    # If test_mode is enabled, override max_chunks
    if test_mode:
        logging.info("Test mode enabled: Limited data processing")
    else:
        logging.info("Test mode disabled: Processing all data")

    # Spawn processes for each GPU
    if test_mode:
        mp.spawn(
            process_on_gpu, 
            args=(world_size, model_name, available_gpus, dataset_path, kmer_size, threshold, max_chunks), 
            nprocs=world_size, 
            join=True
        )
    else:
        mp.spawn(
            process_on_gpu, 
            args=(world_size, model_name, available_gpus, dataset_path, kmer_size, threshold, None), 
            nprocs=world_size, 
            join=True
        )

    combined_edges = []
    combined_weights = []
    combined_nodes = set()
    combined_node_info = {}

    # Aggregate results from all GPUs
    for rank in range(world_size):
        edge_file = f'results_{rank}.json'
        if os.path.exists(edge_file):
            with open(edge_file, 'r') as f:
                data = json.load(f)
                combined_edges.extend(data['edges'])
                combined_weights.extend(data['weights'])
                combined_nodes.update(data['unique_nodes'])
                combined_node_info.update({int(k): v for k, v in data['node_info'].items()})
            os.remove(edge_file)  # Clean up individual files
            logging.info(f"Main: Aggregated data from GPU {rank}")
        else:
            logging.warning(f"Main: {edge_file} does not exist.")

    # Create a mapping from node IDs to unique indices
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(sorted(combined_nodes))}
    logging.info(f"Main: Total unique nodes collected: {len(node_id_to_index)}")

    # Map edges to indices
    mapped_edges = []
    mapped_weights = []
    for (node_i, node_j), weight in zip(combined_edges, combined_weights):
        idx_i = node_id_to_index[node_i]
        idx_j = node_id_to_index[node_j]
        mapped_edges.append((idx_i, idx_j))
        mapped_weights.append(weight)

    # Create a scipy sparse COO matrix
    if mapped_edges:
        row_indices = [edge[0] for edge in mapped_edges]
        col_indices = [edge[1] for edge in mapped_edges]
        data = mapped_weights

        adjacency_matrix = sp.coo_matrix((data, (row_indices, col_indices)), 
                                         shape=(len(node_id_to_index), len(node_id_to_index)),
                                         dtype=float)

        # Since the graph is undirected, add the transpose
        adjacency_matrix = adjacency_matrix + adjacency_matrix.transpose()

        # Remove duplicate entries by converting to CSR and back to COO
        adjacency_matrix = adjacency_matrix.tocsr()
        adjacency_matrix.eliminate_zeros()
        adjacency_matrix = adjacency_matrix.tocoo()

        # Save the adjacency matrix to a file
        sp.save_npz(saved_matrix_fn, adjacency_matrix)
        logging.info(f"Main: Combined adjacency matrix saved to {saved_matrix_fn}")
    else:
        logging.info("Main: No edges to construct the adjacency matrix.")

    # Save node information in the desired format: "node_index": {"string": "...", "position": "start-end"}
    if node_id_to_index:
        node_info_formatted = {}
        for node_id, idx in node_id_to_index.items():
            info = combined_node_info.get(node_id, {'string': 'UNKNOWN', 'position': 'UNKNOWN'})
            node_info_formatted[str(idx)] = {
                'string': info['string'],
                'position': info['position']
            }
        with open(node_info_fn, 'w') as f:
            json.dump(node_info_formatted, f, indent=4)
        logging.info(f"Main: Combined node information saved to {node_info_fn}")
    else:
        logging.info("Main: No node information to save.")

    # Print graph statistics
    print_graph_statistics(combined_edges, combined_nodes)

    logging.info("Script completed successfully.")
    print("Script completed successfully.")

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
