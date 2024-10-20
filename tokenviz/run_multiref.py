import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import gc
import scipy.sparse as sp
import json
import pysam
import os
import re
import logging
import traceback
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import schedule
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np


def clean_sequence(seq):
    # Create translation table once (outside the function if possible)
    translation_table = str.maketrans({'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G'})
    # Use translate instead of regex
    return seq.upper().translate(translation_table)


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
    logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    logging.info('====================================\n')

def setup_logging(log_dir=None):
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'main.log')
    else:
        log_path = 'main.log'

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def tokenize_and_map(tokenizer, sequences, offsets):
    """
    Tokenize sequences with padding and map tokens back to positions in the original sequences.

    Returns:
        List of dictionaries, one per sequence, each containing:
            - 'input_ids': Tensor of input IDs for the sequence.
            - 'tokens': List of tokens.
            - 'token_positions': List of tuples (token_id, token, start, end) positions for each token.
    """
    # Tokenize all sequences at once with padding
    encoded = tokenizer(
        sequences,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors='pt',
        padding=True,  # Pad sequences to the same length
        truncation=True  # Optional: truncate sequences that are too long
    )

    input_ids_batch = encoded['input_ids']  # Tensor of shape [batch_size, max_seq_length]
    offsets_mapping_batch = encoded['offset_mapping']  # List of offset mappings per sequence

    all_mappings = []

    for i, (seq, offset) in enumerate(zip(sequences, offsets)):
        input_ids = input_ids_batch[i]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        offsets_mapping = offsets_mapping_batch[i]

        token_positions = []
        for token_id, token, (start_offset, end_offset) in zip(input_ids, tokens, offsets_mapping):
            # Check for padding tokens
            if token_id.item() == tokenizer.pad_token_id:
                continue  # Skip padding tokens

            # Adjust the positions by the sequence's offset
            token_start = offset + start_offset
            token_end = offset + end_offset

            # Skip [UNK] tokens
            if token == '[UNK]':
                continue

            # Clean the token
            clean_token_str = token.lstrip('Ġ').lstrip('##').lstrip('▁')

            # Verify that the token matches the sequence substring
            sequence_fragment = seq[start_offset:end_offset]
            if sequence_fragment.upper() != clean_token_str.upper():
                logging.warning(f"Token '{token}' does not match sequence at positions {token_start}-{token_end}: '{sequence_fragment}'")

            # Append token and its positions
            token_positions.append((token_id.item(), token, token_start, token_end))

        mapping = {
            'input_ids': input_ids,
            'tokens': tokens,
            'token_positions': token_positions
        }

        all_mappings.append(mapping)

    return all_mappings


def run_model(model, input_ids, layer_num=-1):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True, output_attentions=True)
        
        if isinstance(outputs, tuple):
            attention_weights, last_hidden_state = outputs[-1], outputs[-2]
        else:
            attention_weights, last_hidden_state = outputs.attentions, outputs.last_hidden_state

        # Apply softmax to attention weights
        if isinstance(attention_weights, (list, tuple)):
            # Apply softmax to each tensor in the list
            attention_weights = [
                torch.softmax(attn, dim=-1) for attn in attention_weights
            ]
        else:
            # If it's a tensor, apply softmax directly
            attention_weights = torch.softmax(attention_weights, dim=-1)

    return attention_weights, last_hidden_state

def create_graph(batch_token_positions, batch_attn_weights, threshold=0.01, pad_token_id=None):
    edges = []
    weights = []
    node_info = {}  # mapping node_id -> {'string': ..., 'position': ...}
    unique_node_ids = set()

    for i, (token_positions, attn_weights) in enumerate(zip(batch_token_positions, batch_attn_weights)):
        node_ids = []
        valid_indices = []
        for idx, (token_id, token, start, end) in enumerate(token_positions):
            # Skip padding tokens
            if token_id == pad_token_id:
                continue  # Skip padding tokens
            if token == "[UNK]":
                continue  # Skip [UNK] tokens
            node_ids.append(start)
            valid_indices.append(idx)
            unique_node_ids.add(int(start))  # Ensure start is a native Python int
            # Collect node info
            node_info[int(start)] = {
                'string': token,
                'position': f"{int(start)+1}-{int(end)}"
            }

        seq_len = len(node_ids)
        if seq_len == 0:
            logging.warning(f"Sample {i}: Sequence length is zero after excluding padding and [UNK], skipping graph creation.")
            continue  

        # Filter attention weights to exclude padding tokens
        attn_weights_filtered = [layer[:, valid_indices, :][:, :, valid_indices] for layer in attn_weights]

        # Stack attention weights across layers: [num_layers, num_heads, seq_len, seq_len]
        attn_stack = torch.stack(attn_weights_filtered)  # [num_layers, num_heads, seq_len, seq_len]

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
            continue  

        # Collect edges and weights
        for idx_i, idx_j, weight in zip(valid_edges[0], valid_edges[1], valid_weights):
            node_i = int(node_ids[idx_i.item()])  
            node_j = int(node_ids[idx_j.item()])  # Convert to native Python int
            edges.append((node_i, node_j))
            weights.append(float(weight.item()))  # Ensure weight is a native Python float

    # Return edges, weights, node_info, and unique node_ids
    logging.debug(f"Graph creation completed. Total edges: {len(edges)}, Total nodes: {len(unique_node_ids)}")
    return edges, weights, node_info, unique_node_ids

def print_graph_statistics(total_edges, total_nodes):
    logging.info("\nGraph Statistics Report:")
    logging.info("-------------------------")
    logging.info(f"Number of nodes: {len(total_nodes)}")
    logging.info(f"Number of edges: {len(total_edges)}")
    logging.info("-------------------------")

class DNADataset(Dataset):
    def __init__(self, file_path, chunk_size, max_chunks=None, limit_refs=None):
        """
        Initialize the DNADataset.

        Args:
            file_path (str): Path to the FASTA file.
            chunk_size (int): Size of each chunk to process.
            max_chunks (int, optional): Maximum number of chunks to process for testing.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.limit_refs = limit_refs
        self.fasta = pysam.FastaFile(self.file_path)
        self.references = self.fasta.references
        self.lengths = self.fasta.lengths
        self.chunk_indices = self._compute_chunk_indices(max_chunks)

    def _compute_chunk_indices(self, max_chunks):
        total_chunks = []
        references = self.references
        lengths = self.lengths
        if self.limit_refs is not None:
            references = references[:self.limit_refs]
            lengths = lengths[:self.limit_refs]
            logging.info(f"Limiting to first {self.limit_refs} references.")
        for ref, length in zip(references, lengths):
            num_chunks = (length + self.chunk_size - 1) // self.chunk_size
            indices = [(ref, i * self.chunk_size) for i in range(num_chunks)]
            total_chunks.extend(indices)
        # If max_chunks is set, limit the number of chunks
        if max_chunks is not None:
            total_chunks = total_chunks[:max_chunks]
            logging.info(f"Limited to first {max_chunks} chunks for testing.")
        return total_chunks

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        ref_name, start = self.chunk_indices[idx]
        end = start + self.chunk_size
        seq_length = self.fasta.get_reference_length(ref_name)
        if end > seq_length:
            end = seq_length  # Adjust end if it exceeds the sequence length
        sequence = self.fasta.fetch(ref_name, start, end)

        logging.debug(f"Dataset: Fetched sequence from {ref_name}:{start}-{end}, Length: {len(sequence)}")
        logging.debug(f"Dataset: Sequence content (first 50 bases): {sequence[:50]}")

        # Return ref_name along with sequence and start
        return ref_name, sequence, start

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

class DNABertModule(pl.LightningModule):
    def __init__(self, model_name, kmer_size, threshold):
        super().__init__()
        self.model_name = model_name
        self.kmer_size = kmer_size
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.config = BertConfig.from_pretrained(model_name, return_dict=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=self.config)
        self.max_seq_length = self.model.config.max_position_embeddings if self.model.config.max_position_embeddings else 512

    def predict_step(self, batch, batch_idx):
        ref_names, chunks, chunk_starts = batch

        # Ensure chunk_starts is on CPU
        if chunk_starts.is_cuda:
            chunk_starts = chunk_starts.cpu()

        chunk_starts = chunk_starts.tolist()
        chunks = [clean_sequence(chunk) for chunk in chunks]

        try:
            # Call tokenize_and_map with the correct number of arguments
            token_mappings = tokenize_and_map(
                self.tokenizer,
                chunks,
                chunk_starts
            )

            # Collect input IDs and token positions
            input_ids_list = [mapping['input_ids'] for mapping in token_mappings]
            all_token_positions = [mapping['token_positions'] for mapping in token_mappings]

            # Stack input IDs to create a batch
            input_ids = torch.stack(input_ids_list).to(self.device)

            with torch.cuda.amp.autocast():
                attention_weights, _ = run_model(self.model.to(self.device), input_ids)

            # Reorganize attention_weights per sample
            batch_size = input_ids.size(0)
            attention_weights_per_sample = [[] for _ in range(batch_size)]
            for layer_idx, layer_attn in enumerate(attention_weights):
                if layer_attn.size(0) != batch_size:
                    continue  # Skip this layer
                for i in range(batch_size):
                    attention_weights_per_sample[i].append(layer_attn[i])

            # Process the batch
            edges, weights, node_info, unique_nodes = create_graph(
                all_token_positions, attention_weights_per_sample, threshold=self.threshold, pad_token_id=self.tokenizer.pad_token_id
            )
            return {
                'ref_names': ref_names,
                'edges': edges,
                'weights': weights,
                'node_info': node_info,
                'unique_nodes': unique_nodes
            }

        except Exception as e:
            logging.error(f"Error in predict_step: {e}")
            logging.error(traceback.format_exc())
            return {
                'ref_names': ref_names,
                'edges': [],
                'weights': [],
                'node_info': {},
                'unique_nodes': set()
            }

class DNADataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, chunk_size, batch_size, num_workers, max_chunks=None, limit_refs=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_chunks = max_chunks
        self.limit_refs = limit_refs

    def setup(self, stage=None):
        self.dataset = DNADataset(self.dataset_path, chunk_size=self.chunk_size, max_chunks=self.max_chunks, limit_refs=self.limit_refs)

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=custom_collate_fn,  
        )
    
def custom_collate_fn(batch: List[Tuple[str, str, int]]):
    """
    Custom collate function to handle batches containing reference names.

    Args:
        batch (List[Tuple[str, str, int]]): List of tuples containing (ref_name, sequence, start).

    Returns:
        Tuple[List[str], List[str], torch.Tensor]: Batched reference names, sequences, and starts.
    """
    ref_names, sequences, starts = zip(*batch)
    # Convert starts to a tensor
    starts = torch.tensor(starts, dtype=torch.long)
    return list(ref_names), list(sequences), starts

def main():
    parser = argparse.ArgumentParser(description="Process DNA sequences with DNABERT2 and extract attention graphs.")

    # Required arguments
    parser.add_argument('--kmer_size', type=int, default=6, help='Size of the k-mer (default: 6)')
    parser.add_argument('--saved_matrix_fn', type=str, default="attention_graph_nothreshold.npz", help='Filename to save the adjacency matrix (default: attention_graph_nothreshold.npz)')
    parser.add_argument('--node_info_fn', type=str, default="node_info.json", help='Filename to save node information (default: node_info.json)')
    parser.add_argument('--original_gpus', type=int, nargs='+', default=[0,1], help='List of GPU IDs to use (default: [0,1])')
    parser.add_argument('--model_name', type=str, default="jaandoui/DNABERT2-AttentionExtracted", help='Name of the pre-trained model (default: jaandoui/DNABERT2-AttentionExtracted)')
    parser.add_argument('--dataset_path', type=str, default="hg38.fa", help='Path to the FASTA dataset (default: hg38.fa)')

    # Optional arguments (suggestions)
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold for attention weights to create edges (default: 0.01)')
    parser.add_argument('--node_degree_threshold', type=int, default=1, help='Minimum degree for nodes to be included in the graph (default: 1)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for DataLoader (default: 4)')
    parser.add_argument('--test_mode', action='store_true', help='Enable test mode with limited data processing')
    parser.add_argument('--max_chunks', type=int, default=100, help='Maximum number of chunks to process in test mode (default: 10)')
    parser.add_argument('--limit_refs', type=int, default=2, help='Limit processing to the first N references in test mode (default: 2)')

    parser.add_argument('--log_dir', type=str, default=None, help='Directory to save log files (default: current directory)')
    parser.add_argument('--enable_profiler', action='store_true', help='Enable PyTorch Profiler')

    args = parser.parse_args()

    # Extract arguments
    kmer_size = args.kmer_size
    saved_matrix_fn = args.saved_matrix_fn
    node_info_fn = args.node_info_fn
    original_gpus = args.original_gpus
    model_name = args.model_name
    dataset_path = args.dataset_path
    threshold = args.threshold
    node_degree_threshold = args.node_degree_threshold
    batch_size = args.batch_size
    num_workers = args.num_workers
    test_mode = args.test_mode
    max_chunks = args.max_chunks if test_mode else None
    limit_refs = args.limit_refs if test_mode else None
    log_dir = args.log_dir

    # Setup logging
    setup_logging(log_dir=log_dir)
    logging.getLogger().setLevel(logging.INFO)  # Ensure global logger is INFO
    logging.info("Starting main process")
    
    print_system_info()
    clean_gpu()

    # Initialize the model
    model = DNABertModule(model_name, kmer_size, threshold)

    # Calculate the chunk size based on the model's max sequence length
    chunk_size = model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)
    logging.info(f"Setting chunk size to {chunk_size} based on model's max sequence length.")

    # Initialize the DataModule
    data_module = DNADataModule(dataset_path, chunk_size, batch_size, num_workers, max_chunks=max_chunks, limit_refs=limit_refs)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize the Profiler
    if args.enable_profiler:
        profiler = PyTorchProfiler(
            filename='profile_output',
            schedule=schedule(wait=2, warmup=2, active=3, repeat=1),
            use_cuda=True,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    else:
        profiler = None  # No profiler

    # Initialize the Trainer
    trainer = pl.Trainer(
        devices=len(original_gpus),
        accelerator='gpu',
        strategy='ddp',  # Distributed Data Parallel
        precision=16,
        logger=False,
        profiler=profiler,
    )

    # Run the prediction
    outputs = trainer.predict(model, datamodule=data_module)

    # Initialize dictionaries to hold data per reference
    per_ref_edges = {ref: [] for ref in data_module.dataset.references}
    per_ref_weights = {ref: [] for ref in data_module.dataset.references}
    per_ref_nodes = {ref: set() for ref in data_module.dataset.references}
    per_ref_node_info = {ref: {} for ref in data_module.dataset.references}

    # Aggregate results per reference
    for output_batch in tqdm(outputs, desc="Aggregating outputs"):
        ref_names = output_batch['ref_names']
        edges = output_batch['edges']
        weights = output_batch['weights']
        node_info = output_batch['node_info']
        unique_nodes = output_batch['unique_nodes']

        for ref, edge, weight in zip(ref_names, edges, weights):
            per_ref_edges[ref].append(edge)
            per_ref_weights[ref].append(weight)
            per_ref_node_info[ref].update(node_info)
            per_ref_nodes[ref].update(unique_nodes)

    # Create directories for saving outputs
    adj_matrix_dir = os.path.join(os.path.dirname(saved_matrix_fn), "adjacency_matrices")#/NHEK_dev")
    node_info_dir = os.path.join(os.path.dirname(node_info_fn), "node_info")#/NHEK_dev")

    os.makedirs(adj_matrix_dir, exist_ok=True)
    os.makedirs(node_info_dir, exist_ok=True)

    references = data_module.dataset.references
    if limit_refs is not None:
        references = references[:limit_refs]
        logging.info(f"Processing only the first {limit_refs} references.")
    else:
        references = data_module.dataset.references

    # Process each reference separately
    for ref in references:
        logging.info(f"Processing reference: {ref}")

        combined_edges = per_ref_edges[ref]
        combined_weights = per_ref_weights[ref]
        combined_node_info = per_ref_node_info[ref]

        if not combined_edges:
            logging.warning(f"No edges found for reference {ref}, skipping.")
            continue

        # Flatten the list of edges and weights
        edges = combined_edges  # List of tuples (node_i, node_j)
        weights = combined_weights  # List of corresponding weights

        # Initialize nodes_to_include with all nodes present in edges
        nodes_to_include = set()
        for edge in edges:
            nodes_to_include.update(edge)

        # Begin iterative process
        while True:
            # Compute node degrees
            node_degrees = {node_id: 0 for node_id in nodes_to_include}
            for (node_i, node_j) in edges:
                node_degrees[node_i] += 1
                node_degrees[node_j] += 1

            # Filter nodes based on degree threshold
            nodes_meeting_threshold = {node_id for node_id, degree in node_degrees.items() if degree >= node_degree_threshold}

            # If no changes in nodes_to_include, break the loop
            if nodes_meeting_threshold == nodes_to_include:
                break  # Degrees meet threshold, proceed

            # Update nodes_to_include
            nodes_to_include = nodes_meeting_threshold

            # Filter edges and weights
            filtered_edges = []
            filtered_weights = []
            for (edge, weight) in zip(edges, weights):
                node_i, node_j = edge
                if node_i in nodes_to_include and node_j in nodes_to_include:
                    filtered_edges.append(edge)
                    filtered_weights.append(weight)

            # Update edges and weights for next iteration
            edges = filtered_edges
            weights = filtered_weights

        # Proceed with nodes_to_include and edges/weights
        logging.info(f"[{ref}] Nodes after iterative filtering: {len(nodes_to_include)}")

        # Create a mapping from node IDs to indices
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(sorted(nodes_to_include))}
        logging.info(f"Total unique nodes collected for {ref}: {len(node_id_to_index)}")

        # Map edges
        mapped_edges = []
        mapped_weights = []
        for (node_i, node_j), weight in zip(edges, weights):
            idx_i = node_id_to_index[node_i]
            idx_j = node_id_to_index[node_j]
            mapped_edges.append((idx_i, idx_j))
            mapped_weights.append(weight)

        if mapped_edges:
            # Create a scipy sparse COO matrix
            row_indices = [edge[0] for edge in mapped_edges]
            col_indices = [edge[1] for edge in mapped_edges]
            data = mapped_weights

            adjacency_matrix = sp.coo_matrix(
                (data, (row_indices, col_indices)),
                shape=(len(node_id_to_index), len(node_id_to_index)),
                dtype=float
            )

            # Since the graph is undirected, add the transpose
            adjacency_matrix = adjacency_matrix + adjacency_matrix.transpose()

            # Remove duplicate entries by converting to CSR and eliminating zeros
            adjacency_matrix = adjacency_matrix.tocsr()
            adjacency_matrix.eliminate_zeros()

            # Save the adjacency matrix to a file specific to the reference
            base_saved_matrix_fn = os.path.basename(saved_matrix_fn)
            matrix_fn = os.path.join(adj_matrix_dir, f"{ref}_{base_saved_matrix_fn}")

            sp.save_npz(matrix_fn, adjacency_matrix)
            logging.info(f"[{ref}] Adjacency matrix saved to {matrix_fn}")

            # Compute weighted degrees (sum of edge weights connected to each node)
            weighted_degrees = adjacency_matrix.sum(axis=1).A1  # Convert to a 1D NumPy array

            # Normalize weighted degrees between 0 and 1 for color mapping
            if weighted_degrees.max() != weighted_degrees.min():
                weighted_degree_norm = (weighted_degrees - weighted_degrees.min()) / (weighted_degrees.max() - weighted_degrees.min())
            else:
                # If all weighted degrees are the same, set all normalized values to 0.5
                weighted_degree_norm = np.full_like(weighted_degrees, 0.5)

            # Choose a colormap
            cmap = plt.get_cmap('viridis')  # You can choose any colormap you like
            scalarMap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap=cmap)

            # Map normalized weighted degrees to RGBA colors
            node_colors = scalarMap.to_rgba(weighted_degree_norm)

            # Convert RGBA colors to HEX format for JSON serialization
            node_colors_hex = [colors.rgb2hex(color) for color in node_colors]

            # Save node information including 'weighted_degree' and 'color'
            if node_id_to_index:
                node_info_formatted = {}
                for node_id, idx in node_id_to_index.items():
                    info = combined_node_info.get(node_id, {'string': 'UNKNOWN', 'position': 'UNKNOWN'})
                    node_info_formatted[str(idx)] = {
                        'string': info['string'],
                        'position': info['position'],
                        'weighted_degree': float(weighted_degrees[idx]),
                        'color': node_colors_hex[idx]
                    }

                base_node_info_fn = os.path.basename(node_info_fn)
                node_info_fn_ref = os.path.join(node_info_dir, f"{ref}_{base_node_info_fn}")

                with open(node_info_fn_ref, 'w') as f:
                    json.dump(node_info_formatted, f, indent=4)
                logging.info(f"[{ref}] Node information saved to {node_info_fn_ref}")
            else:
                logging.info(f"[{ref}] No node information to save.")

            # Print graph statistics for this reference
            print_graph_statistics(mapped_edges, node_id_to_index.keys())

        else:
            logging.info(f"[{ref}] No edges to construct the adjacency matrix after filtering.")
            continue  # Skip to the next reference

    # Export profiler summary to a text file
    if args.enable_profiler:
        profiler_summary = profiler.summary()
        with open('profiler_output.txt', 'w') as f:
            f.write(profiler_summary)
        logging.info('Profiler summary saved to profiler_output.txt')

    logging.info("Script completed successfully.")
    print(f"Script completed successfully. Threshold was: {threshold}")

if __name__ == "__main__":
    main()
