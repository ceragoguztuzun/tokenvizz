import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import gc
import networkx as nx
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

def setup_logging(rank):
    if isinstance(rank, int):
        log_filename = f'process_{rank}.log'
        rankname = f'GPU {rank}'
    else:
        log_filename = 'main.log'
        rankname = str(rank)

    # Create a logger specific to the process
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler
    fh = logging.FileHandler(log_filename, mode='w')
    fh.setLevel(logging.DEBUG)

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

def tokenize_and_map(tokenizer, ref_dna_chunks, device, max_seq_length, offsets):
    tokenized = tokenizer(
        list(ref_dna_chunks),
        return_tensors='pt',
        truncation=True,
        max_length=max_seq_length,
        padding=True,
    )
    input_ids = tokenized["input_ids"].to(device)
    all_token_positions = []
    kmer_size = 6  # Adjust based on your tokenizer's k-mer size
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
    G = nx.Graph()
    batch_size = len(batch_token_positions)

    # Prepare node information
    node_ids_list = []
    for token_positions in batch_token_positions:
        node_ids = []
        for (token, start, _) in token_positions:
            node_ids.append(start)
            G.add_node(start, label=token, start_position=start)
        node_ids_list.append(node_ids)

    # Each element in batch_attn_weights is a list of tensors (one per layer) for a sample
    for i in range(batch_size):
        node_ids = node_ids_list[i]
        seq_len = len(node_ids)
        if seq_len == 0:
            logging.warning(f"Sample {i}: Sequence length is zero, skipping graph creation.")
            continue  # Skip this sample

        try:
            # Stack attention weights across layers: [num_layers, num_heads, seq_len, seq_len]
            attn_stack = torch.stack(batch_attn_weights[i])  # [num_layers, num_heads, seq_len, seq_len]

            # Average over layers and heads: [seq_len, seq_len]
            avg_attn_matrix = attn_stack.mean(dim=[0, 1])  # Shape: [seq_len, seq_len]
            logging.debug(f"Sample {i}: Averaged attention matrix shape: {avg_attn_matrix.shape}")

            # Compute edge indices
            edge_indices = torch.triu_indices(seq_len, seq_len, offset=1).to(avg_attn_matrix.device)
            edge_weights = avg_attn_matrix[edge_indices[0], edge_indices[1]]

            # Apply threshold
            mask = edge_weights >= threshold
            valid_edges = edge_indices[:, mask]
            valid_weights = edge_weights[mask]

            logging.debug(f"Sample {i}: Number of valid edges after thresholding: {valid_edges.shape[1]}")

            if valid_edges.shape[1] == 0:
                logging.warning(f"Sample {i}: No valid edges after thresholding.")
                continue  # Skip this sample

            # Add edges
            for idx_i, idx_j, weight in zip(valid_edges[0], valid_edges[1], valid_weights):
                node_i = node_ids[idx_i.item()]
                node_j = node_ids[idx_j.item()]
                G.add_edge(node_i, node_j, weight=weight.item())

        except Exception as e:
            logging.error(f"Sample {i}: Error during graph creation: {e}")
            logging.error(traceback.format_exc())
            continue  # Skip this sample in case of error

    return G

def print_graph_statistics(G):
    logging.info("\nGraph Statistics Report:")
    logging.info("-------------------------")
    logging.info(f"Number of nodes: {G.number_of_nodes()}")
    logging.info(f"Number of edges: {G.number_of_edges()}")
    logging.info("-------------------------")

class DNADataset(Dataset):
    def __init__(self, file_path, rank, world_size, chunk_size):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size
        self.fasta = pysam.FastaFile(self.file_path)
        self.references = self.fasta.references
        self.lengths = self.fasta.lengths
        self.chunk_indices = self._compute_chunk_indices()

    def _compute_chunk_indices(self):
        total_chunks = []
        for ref, length in zip(self.references, self.lengths):
            num_chunks = (length + self.chunk_size - 1) // self.chunk_size
            indices = [(ref, i * self.chunk_size) for i in range(num_chunks)]
            total_chunks.extend(indices)
        # Distribute indices among processes
        distributed_chunks = total_chunks[self.rank::self.world_size]
        logging.info(f"GPU {self.rank}: Assigned {len(distributed_chunks)} chunks.")
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

def process_on_gpu(rank, world_size, model_name, available_gpus):
    setup_logging(rank)
    gpu_id = available_gpus[rank]
    device = torch.device(f"cuda:{gpu_id}")
    logging.info(f"Process {rank} is using GPU {gpu_id}")

    model, tokenizer, max_seq_length = load_model_and_tokenizer(model_name, device)
    chunk_size = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    logging.info(f"Setting chunk size to {chunk_size} based on model's max sequence length.")

    dataset = DNADataset("/home/cxo147/ceRAG_viz/data/hg38.fa", rank, world_size, chunk_size)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4, drop_last=False)  # Increased batch_size

    G = nx.Graph()
    node_info = {}

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"GPU {rank} - Processing chunks", position=rank)):
        chunks, chunk_starts = batch
        chunks = [clean_sequence(chunk) for chunk in chunks]
        chunk_starts = chunk_starts.numpy()
        try:
            input_ids, all_token_positions = tokenize_and_map(
                tokenizer, chunks, device, max_seq_length, chunk_starts
            )

            logging.debug(f"Batch {batch_idx}: Token positions batch size: {len(all_token_positions)}")

            # Run the model
            with torch.cuda.amp.autocast():
                attention_weights, _ = run_model(model, input_ids)

            logging.debug(f"Batch {batch_idx}: Number of attention layers: {len(attention_weights)}")
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

            logging.debug(f"Batch {batch_idx}: Attention weights per sample: {len(attention_weights_per_sample)} samples")

            # Ensure the batch size matches
            if len(all_token_positions) != len(attention_weights_per_sample):
                logging.error(f"Batch {batch_idx}: Mismatch between token positions and attention matrices: {len(all_token_positions)} vs {len(attention_weights_per_sample)}")
                continue  # Skip this batch

            # Process the batch
            chunk_G = create_graph(all_token_positions, attention_weights_per_sample)
            G = nx.compose(G, chunk_G)
        except Exception as e:
            logging.error(f"GPU {rank} - Error occurred: {e}")
            logging.error(traceback.format_exc())
            continue  # Skip this batch

    logging.info(f"Saving results for GPU {rank}")
    torch.save((G, node_info), f'results_{rank}.pt')

def main():
    setup_logging('MAIN')
    logging.info("Starting main process")
    print_system_info()
    clean_gpu()

    # Specify the GPUs you want to use
    original_gpus = [0, 4, 5, 6, 7] 
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, original_gpus))
    logging.info(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Adjust GPU indices after setting CUDA_VISIBLE_DEVICES
    available_gpus = list(range(len(original_gpus)))
    world_size = len(available_gpus)
    logging.info(f"Adjusted available GPUs: {available_gpus}")

    model_name = "jaandoui/DNABERT2-AttentionExtracted"

    mp.spawn(process_on_gpu, args=(world_size, model_name, available_gpus), nprocs=world_size, join=True)

    combined_G = nx.Graph()
    combined_node_info = {}

    for rank in range(world_size):
        G, node_info = torch.load(f'results_{rank}.pt')
        combined_G.update(G)
        combined_node_info.update(node_info)

    print_graph_statistics(combined_G)

    with open('node_info.json', 'w') as f:
        json.dump(combined_node_info, f)
    logging.info("Combined node information saved to node_info.json")

    saved_G_fn = "attention_graph_nothreshold"
    nx.write_edgelist(combined_G, f"{saved_G_fn}.csv", delimiter=",", data=['weight'])
    logging.info(f"Combined graph saved to {saved_G_fn}.csv")

    logging.info("Script completed successfully.")

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
