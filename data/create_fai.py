import pysam
import sys
import os

def create_fai(fasta_path):
    """
    Generates a .fai index file for the given FASTA file using pysam.
    """
    if not os.path.isfile(fasta_path):
        print(f"Error: FASTA file '{fasta_path}' does not exist.")
        return
    
    try:
        pysam.faidx(fasta_path)
        print(f"Index created for '{fasta_path}'.")
    except Exception as e:
        print(f"Error creating index for '{fasta_path}': {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_fai.py <fasta_file1.fa> [<fasta_file2.fa> ...]")
        sys.exit(1)
    
    fasta_files = sys.argv[1:]
    for fasta in fasta_files:
        create_fai(fasta)
