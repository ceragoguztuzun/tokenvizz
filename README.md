
# Tokenvizz User Guide

## Introduction

**Tokenvizz** is a simple yet powerful tool that allows you to visualize genomic sequences by converting them into tokens and displaying their relationships through attention graphs. This guide will help you set up the tool, process your data, and explore the interactive visualization in your web browser.

https://github.com/user-attachments/assets/8e219afd-a471-4d28-b5ef-6f4e4f8d8c9b


## Overview

To use DNA Tokenviz, you'll follow these main steps:

1.  **Set Up the Environment**
2.  **Tokenize DNA Sequences and Generate Attention Graphs**
3.  **Visualize the Attention Graphs with Text Highlights**

Let's get started!

----------

## Step 1: Set Up the Environment
    
**Create and activate the environment**
    
   Open your terminal and run:
    
```bash
conda env create -f dna.yml
```
```bash
conda activate dna
``` 
----------

## Step 2: Tokenize DNA Sequences and Generate Attention Graphs

In this step, you'll convert your DNA sequences into tokens and create attention graphs that show how different parts of the sequences relate to each other.

### What You Need

-   **Dataset Path**: The path to your DNA sequences in a `.fa` (FASTA) file. Ensure the `.fai` index file is in the same directory.
-   **Model Name**: The name of the pre-trained model from Hugging Face to use (e.g., [`jaandoui/DNABERT2-AttentionExtracted`](https://huggingface.co/jaandoui/DNABERT2-AttentionExtracted/discussions), [`InstaDeepAI/nucleotide-transformer-2.5b-multi-species`](https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species)).
-   **Optional Parameters**: Additional settings you can adjust (explained below).

Run the following command in your terminal:
```bash
python3 tokenize.py \
    --dataset_path /path/to/your/data.fa \
    --model_name jaandoui/DNABERT2-AttentionExtracted \
    --saved_matrix_fn /path/to/save/attention_graph.npz \
    --node_info_fn /path/to/save/node_info.json 
```

### What Happens Next

The script processes the inputted DNA sequences and generates:
-   **Adjacency Matrices**: Saved as `.npz` files, containing the attention scores between tokens.
-   **Node Information**: Saved as `.json` files, containing details about each token.
----------

## Step 3: Visualize the Attention Graphs with Text Highlights

Now, you'll create an interactive graph with text highlights that you can view in your web browser.

### What You Need

-   **Output File**: The name of the HTML file that will contain your visualization.
-   **Edges File**: The `.npz` file with the adjacency matrices you generated earlier.
-   **Node Info File**: The `.json` file with node information you generated earlier.
-   **FASTA File**: The path to your original `.fa` file for text highlights.

#### 1. Generate the Visualization HTML

Run the following command after replacing the paths with the locations of your files:
```bash
python3 viz.py \
    --model_name "jaandoui/DNABERT2-AttentionExtracted" \
    --edges_file "/path/to/your/adjacency_matrices/{reference_name}_attention_graph.npz" \
    --node_info_file "/path/to/your/node_info/{reference_name}_node_info.json" \
    --output_file "graph_visualization.html"
```
`{reference_name}` is the name of the reference from the FASTA file you want to visualize interactively.

#### 2. (Optional) Start the Flask Server for Text Highlights
To enable text highlighting features in your visualization, run the `server.py` script with the `--fasta` argument.
```bash
python3 server.py --fasta '/path/to/your/data.fa'` 
```
-   Replace `'/path/to/your/data.fa'` with the path to your original `.fa` file.

The Flask server will start running to provide the necessary backend for text highlights.

#### 3. Serve the HTML File Locally
To properly load the visualization and interact with the Flask server, start a local HTTP server in the directory containing your `graph_visualization.html` file.

Open a new terminal window and run:
```bash
python3 -m http.server 8000 
```
This command starts a simple HTTP server on port `8000`.

#### 4. Open the Visualization in Your Browser
-   Open your web browser and navigate to: `http://localhost:8000/graph_visualization.html`
-   The visualization should now load, and the text highlighting features provided by the Flask server will be available.
-   You can adjust edge weight thresholds and search for specific positions within the DNA sequences.
- You can click on nodes to get mode information on them.

----------

## Understanding the Command Arguments

### For `tokenize.py`

-   `--kmer_size`: Size of the k-mer for tokenization (default: `6`).
-   `--saved_matrix_fn`: Filename to save the adjacency matrix (default: `"attention_graph_nothreshold.npz"`).
-   `--node_info_fn`: Filename to save node information (default: `"node_info.json"`).
-   `--original_gpus`: GPUs to use (default: `[0,1]`).
-   `--model_name`: Pre-trained model name from Hugging Face (default: `"jaandoui/DNABERT2-AttentionExtracted"`).
-   `--dataset_path`: Path to your `.fa` file (default: `"hg38.fa"`).
-   `--threshold`: Attention score threshold for creating edges (default: `0.01`).
-   `--batch_size`: Number of sequences to process at once (default: `16`).
-   `--num_workers`: Number of worker threads (default: `4`).
-   `--log_dir`: Directory for log files (default: current directory).

### For `viz.py`

-   `--model_name`: Pre-trained model name used earlier.
-   `--edges_file`: Path to the specific reference's `.npz` file with adjacency matrices.
-   `--node_info_file`: Path to the specific reference's`.json` file with node info.
-   `--output_file`: Name for your HTML visualization file (default: `"graph_vizzy.html"`).
-   `--default_weight`: Edge weight threshold for visualization (default: `0.01`).

----------

## Extending this work...
Feel free to expand on this tool by:

-   Using the graph in your downstream applications.
-   Exploring different pre-trained models.
-   Adjusting thresholds for more or less detail in the visualization.
-   Processing larger datasets by tweaking performance parameters.
-   Enhancing the visualization with additional interactive features.

Please contact to share ideas, ask questions and report bugs: cxo147@case.edu

