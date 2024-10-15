import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from Bio import SeqIO  # Requires Biopython

def extract_sequence_positions(fasta_file, seq_full_name):
    """
    Extract the positions of promoter, separator, and enhancer from the concatenated sequence.
    Args:
        fasta_file (str): Path to the multi-FASTA file containing the concatenated sequences.
        seq_full_name (str): The reference name within the FASTA file to extract.
    Returns:
        promoter_start (int), promoter_end (int): Positions of the promoter sequence.
        enhancer_start (int), enhancer_end (int): Positions of the enhancer sequence.
    """
    # Parse the multi-FASTA file and find the specific sequence with seq_full_name
    for record in SeqIO.parse(fasta_file, "fasta"):
        if record.id == seq_full_name:
            sequence = str(record.seq).upper()
            separator = 'N' * 10  # We assume the separator is 'N' repeated 10 times
            separator_start = sequence.find(separator)
            if separator_start == -1:
                print(f"[DEBUG] Separator not found in sequence for {seq_full_name}.")
                return None, None, None, None
            promoter_start = 0
            promoter_end = separator_start
            enhancer_start = separator_start + len(separator)
            enhancer_end = len(sequence)
            return promoter_start, promoter_end, enhancer_start, enhancer_end
    
    print(f"[DEBUG] Sequence {seq_full_name} not found in {fasta_file}.")
    return None, None, None, None

def load_attention_matrix(attention_file):
    """
    Load the attention matrix from the .npz file.
    Args:
        attention_file (str): Path to the attention matrix file.
    Returns:
        attention_matrix (np.ndarray): Dense attention matrix.
    """
    if not os.path.exists(attention_file):
        print(f"[DEBUG] Attention file {attention_file} does not exist.")
        return None
    try:
        attention_sparse = sparse.load_npz(attention_file).tocoo()
        num_nodes = max(attention_sparse.row.max(), attention_sparse.col.max()) + 1
        attention_matrix = sparse.coo_matrix(
            (attention_sparse.data, (attention_sparse.row, attention_sparse.col)),
            shape=(num_nodes, num_nodes)
        ).toarray()
        return attention_matrix
    except Exception as e:
        print(f"[DEBUG] Failed to load attention matrix {attention_file}: {e}.")
        return None

def create_attention_heatmap(attention_matrix, promoter_indices, enhancer_indices, output_path):
    """
    Create and save a heatmap of attention scores between promoter and enhancer regions.
    Args:
        attention_matrix (np.ndarray): Dense attention matrix.
        promoter_indices (list): Indices of promoter nodes.
        enhancer_indices (list): Indices of enhancer nodes.
        output_path (str): Path to save the heatmap image.
    """
    # Extract sub-matrix of attention scores between promoter and enhancer
    sub_matrix = attention_matrix[np.ix_(promoter_indices, enhancer_indices)]
    
    print('sub', sub_matrix)
    print("Max attention score:", np.max(sub_matrix))
    print("Min attention score:", np.min(sub_matrix))
    print("Mean attention score:", np.mean(sub_matrix))

    if sub_matrix.size == 0:
        print(f"[DEBUG] Sub-matrix between promoter and enhancer is empty.")
        return
    
    #normalizing sub_matrix
    if np.max(sub_matrix) > 0:
        sub_matrix = (sub_matrix - np.min(sub_matrix)) / (np.max(sub_matrix) - np.min(sub_matrix))

    plt.figure(figsize=(10, 8))
    sns.heatmap(sub_matrix, cmap='viridis')
    plt.title('Attention Heatmap between Promoter and Enhancer')
    plt.xlabel('Enhancer Tokens')
    plt.ylabel('Promoter Tokens')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Heatmap saved to {output_path}")

def get_node_positions(node_info_file):
    with open(node_info_file, 'r') as f:
        node_info = json.load(f)
    node_positions = {}
    for node_id, info in node_info.items():
        position = info.get('position', '0-0')
        try:
            start, end = map(int, position.split('-'))
            #print(f"Node {node_id}: Start = {start}, End = {end}")  # Debugging: Print node positions
            node_positions[int(node_id)] = (start, end)
        except ValueError:
            node_positions[int(node_id)] = (0, 0)
    return node_positions

def main():
    base_dir = '/usr/homes/cxo147/ceRAG_viz/'  # Update this path
    split = 'test'  # Can be 'train', 'dev', or 'test'
    output_dir = '/usr/homes/cxo147/ceRAG_viz/heatmaps'  # Update this path
    os.makedirs(output_dir, exist_ok=True)

    label_file = os.path.join(base_dir, f"data/NHEK/NHEK_{split}_label_mapping.txt")
    fasta_file = os.path.join(base_dir, f"data/NHEK/NHEK_{split}.fa")  # Multi-FASTA file for all sequences

    with open(label_file, 'r') as f:
        label_lines = f.readlines()

    label_dict = {}
    for line in label_lines:
        if ':' not in line:
            continue
        seq_full_name, label = line.strip().split(':')
        label_dict[seq_full_name.strip()] = int(label.strip())

    for seq_full_name in label_dict.keys():
        # Paths to required files
        attention_file = os.path.join(
            base_dir, 
            f"tokenviz/outputs/adjacency_matrices/NHEK_{split}/{seq_full_name}_attention_graph.npz"
        )
        node_info_file = os.path.join(
            base_dir,
            f"tokenviz/outputs/node_info/NHEK_{split}/{seq_full_name}_node_info.json"
        )
        if not (os.path.exists(attention_file) and os.path.exists(node_info_file)):
            print(f"[DEBUG] Missing files for {seq_full_name}. Skipping.")
            continue

        # Extract promoter and enhancer positions from the multi-FASTA file
        promoter_start, promoter_end, enhancer_start, enhancer_end = extract_sequence_positions(fasta_file, seq_full_name)
        if promoter_start is None:
            continue

        # Load attention matrix
        attention_matrix = load_attention_matrix(attention_file)
        if attention_matrix is None:
            continue

        # Get node positions
        node_positions = get_node_positions(node_info_file)
        # Identify node indices corresponding to promoter, enhancer, and separator regions
        promoter_indices = []
        enhancer_indices = []

        for node_id, (start, end) in node_positions.items():
            # Since node positions are sub-sequences, we'll consider the midpoint
            node_pos = (start + end) // 2

            # Assign nodes to promoter, separator, or enhancer based on their positions
            if promoter_start <= node_pos < promoter_end:
                promoter_indices.append(node_id)
                #print(f"Promoter Node {node_id}: Start = {start}, End = {end}")  # Debugging: Print promoter nodes
            elif enhancer_start <= node_pos < enhancer_end:
                enhancer_indices.append(node_id)
                #print(f"Enhancer Node {node_id}: Start = {start}, End = {end}")  # Debugging: Print enhancer nodes
            else:
                print(f"Node {node_id} falls in the separator or outside range: Start = {start}, End = {end}")

        if not promoter_indices or not enhancer_indices:
            print(f"[DEBUG] No promoter or enhancer nodes found for {seq_full_name}. Skipping.")
            continue

        # Create and save heatmap
        output_path = os.path.join(output_dir, f"{seq_full_name}_attention_heatmap.png")
        create_attention_heatmap(attention_matrix, promoter_indices, enhancer_indices, output_path)

    base_dir = '/usr/homes/cxo147/ceRAG_viz/'  # Update this path
    split = 'test'  # Can be 'train', 'dev', or 'test'
    output_dir = '/usr/homes/cxo147/ceRAG_viz/heatmaps'  # Update this path
    os.makedirs(output_dir, exist_ok=True)

    label_file = os.path.join(base_dir, f"data/NHEK/NHEK_{split}_label_mapping.txt")
    fasta_file = os.path.join(base_dir, f"data/NHEK/NHEK_{split}.fa")  # Multi-FASTA file for all sequences

    with open(label_file, 'r') as f:
        label_lines = f.readlines()

    label_dict = {}
    for line in label_lines:
        if ':' not in line:
            continue
        seq_full_name, label = line.strip().split(':')
        label_dict[seq_full_name.strip()] = int(label.strip())

    for seq_full_name in label_dict.keys():
        # Paths to required files
        attention_file = os.path.join(
            base_dir, 
            f"tokenviz/outputs/adjacency_matrices/NHEK_{split}/{seq_full_name}_attention_graph.npz"
        )
        node_info_file = os.path.join(
            base_dir,
            f"tokenviz/outputs/node_info/NHEK_{split}/{seq_full_name}_node_info.json"
        )
        if not (os.path.exists(attention_file) and os.path.exists(node_info_file)):
            print(f"[DEBUG] Missing files for {seq_full_name}. Skipping.")
            continue

        # Extract promoter and enhancer positions from the multi-FASTA file
        promoter_start, promoter_end, enhancer_start, enhancer_end = extract_sequence_positions(fasta_file, seq_full_name)
        if promoter_start is None:
            continue

        # Load attention matrix
        attention_matrix = load_attention_matrix(attention_file)
        print('attn', attention_matrix)

        if attention_matrix is None:
            continue

        # Get node positions
        node_positions = get_node_positions(node_info_file)
        # Identify node indices corresponding to promoter and enhancer regions
        promoter_indices = []
        enhancer_indices = []
        for node_id, (start, end) in node_positions.items():
            # Since node positions are sub-sequences, we'll consider the midpoint
            node_pos = (start + end) // 2
            if promoter_start <= node_pos < promoter_end:
                promoter_indices.append(node_id)
                print(f"Promoter Node {node_id}: Start = {start}, End = {end}")  # Debugging: Print promoter nodes
            elif enhancer_start <= node_pos < enhancer_end:
                enhancer_indices.append(node_id)
                print(f"Enhancer Node {node_id}: Start = {start}, End = {end}")  # Debugging: Print enhancer nodes


        if not promoter_indices or not enhancer_indices:
            print(f"[DEBUG] No promoter or enhancer nodes found for {seq_full_name}. Skipping.")
            continue

        # Create and save heatmap
        output_path = os.path.join(output_dir, f"{seq_full_name}_attention_heatmap.png")
        create_attention_heatmap(attention_matrix, promoter_indices, enhancer_indices, output_path)

if __name__ == "__main__":
    main()
    print('---done---')
