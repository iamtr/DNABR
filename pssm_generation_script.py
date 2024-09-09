import pandas as pd
import pickle
import argparse
import os
import subprocess
import tempfile
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
from multiprocessing import Pool

def generate_pssm(sequence):
    DB_PATH = "./databases/uniprot_sprot.fasta"
    output_pssm = "output.pssm"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as temp_fasta:
        SeqIO.write([SeqRecord(Seq(sequence))], temp_fasta, "fasta")
        temp_fasta_path = temp_fasta.name

    try:
        subprocess.run(["psiblast", "-query", temp_fasta_path, "-db", DB_PATH,
                        "-out_ascii_pssm", output_pssm, "-num_iterations", "3", "-evalue", "0.001"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    finally:
        os.remove(temp_fasta_path)

    try:
        pssm_df = pd.read_csv(output_pssm, delim_whitespace=True, skiprows=3, header=None)
        os.remove(output_pssm)
        pssm_array = pssm_df.iloc[:-5, 2:22].to_numpy(dtype=int)
        return pssm_array
    except FileNotFoundError:
        print(f"Error: PSSM file not found for sequence {sequence}")
        return None

def pre_generate_pssm(input_df, num_processes=4):
    with Pool(processes=num_processes) as pool:
        pssm_list = pool.map(generate_pssm, input_df['seq'])
    return pssm_list

def save_pssms(pssm_list, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(pssm_list, f)

def main():
    parser = argparse.ArgumentParser(description="Generate PSSMs and save to a file.")
    parser.add_argument("input_csv", type=str, help="Input CSV file path.")
    parser.add_argument("output_pkl", type=str, help="Output pickle file path.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes to use.")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input_csv)
    pssm_list = pre_generate_pssm(input_df, num_processes=args.num_processes)
    save_pssms(pssm_list, args.output_pkl)
    print(f"Generated {len(pssm_list)} PSSMs, saved to {args.output_pkl}")

if __name__ == "__main__":
    main()
