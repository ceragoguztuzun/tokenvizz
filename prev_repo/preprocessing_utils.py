import numpy as np
import pandas as pd
import random
import pysam
import os

# source: https://vincebuffalo.com/notes/2014/01/17/md-tags-in-bam-files.html
def extract_info(read):
    original_sequence = read.query_sequence
    converted_sequence = ''
    if read.has_tag('MD'):
        md_string = read.get_tag('MD')
        current_pos = 0
        skip = 0
        for char in md_string:
            if char.isdigit():
                skip = skip * 10 + int(char)
            elif char.isalpha() or char == '^':
                if skip > 0:
                    converted_sequence += original_sequence[current_pos:current_pos+skip]
                    current_pos += skip
                    skip = 0
                if char.isalpha():
                    converted_sequence += char
                    current_pos += 1
        if skip > 0:
            converted_sequence += original_sequence[current_pos:current_pos+skip]
    return original_sequence, converted_sequence

def encode_methylation(original, bisulfate_converted, YC_tag = 'CT'):
    encoded = []

    if YC_tag == 'CT':
        orig_base = 'C'
        conv_base = 'T'
        enc_base = 'Y'

    elif YC_tag == 'GA':
        orig_base = 'G'
        conv_base = 'A'
        enc_base = 'R'

    for o, b in zip(original, bisulfate_converted):
        if o == orig_base and b == conv_base:
            encoded.append(enc_base) #where unmethylated C becomes T, or G becomes A.
        else:
            encoded.append(o) # keep original base
    return ''.join(encoded)

def get_reads_for_SRR(bam_dir, bam_file):    

    bam_fp = f'{bam_dir}{bam_file}'
    #pysam.index(bam_fp) # uncomment this if bai files are not generated already.
    bam_file = pysam.AlignmentFile(bam_fp, 'rb')
    methylation_data = []

    for read in bam_file.fetch():
        if not read.is_unmapped: #skip unmapped reads
            #read has to have a YC tag, MQ check
            if read.has_tag('YC') and (read.mapping_quality > 20):
                original_sequence, converted_sequence = extract_info(read)
                YC_tag = read.get_tag("YC")

                methylation_data.append((original_sequence, converted_sequence, YC_tag))

    bam_file.close()

    encoded_sequences = []

    for original, bisulfite, YC_tag in methylation_data:
        encoded = encode_methylation(original, bisulfite, YC_tag)
        encoded_sequences.append(encoded)
    
    # downsample
    contains_y_seq = [seq for seq in encoded_sequences if 'Y' in seq]
    contains_r_seq = [seq for seq in encoded_sequences if 'R' in seq]
    final_seq = contains_y_seq + contains_r_seq

    print(f'encoded - {len(contains_y_seq)} have C->T methylation sequences, and {len(contains_r_seq)} G->A.')
    return final_seq
