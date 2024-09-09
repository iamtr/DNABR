import sklearn
import numpy as np
import pandas as pd
from Bio import SeqIO, SeqRecord, Seq
import subprocess
import tempfile
import os
import pfam2go

AMINO_ACID_INDICES = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 
                      'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

PKA_AMINO_GROUP = np.array([9.69, 9.04, 8.80, 9.60, 10.28, 9.13, 9.67, 9.60, 9.17, 9.60,
                            9.60, 8.95, 9.21, 9.13, 10.60, 9.15, 9.10, 9.39, 9.11, 9.62])
PKA_CARBOXYL_GROUP = np.array([2.34, 2.17, 2.02, 1.88, 1.96, 2.17, 2.19, 2.34, 1.82, 2.36,
                               2.36, 2.18, 2.28, 1.83, 1.99, 2.21, 2.09, 2.83, 2.20, 2.32])
EIIP = np.array([0.03731, 0.09593, 0.00359, 0.12630, 0.08292, 0.07606, 0.00580, 0.00499, 0.02415, 0.0000, 
                 0.0000, 0.03710, 0.08226, 0.09460, 0.01979, 0.08292, 0.09408, 0.05481, 0.05159, 0.00569])
LONE_ELECTRON_PAIRS = np.array([0, 0, 1, 2, 1, 1, 2, 0, 1, 0, 
                                0, 0, 0, 0, 0, 1, 1, 0, 1, 0])
WIENER_INDEX = np.array([0.3466, 0.1156, 0.3856, 0.2274, 0.0501, 0.6379, 0.1938, 0.1038, 0.2013,
                       0.2863, 0.1071, 0.7767, 0.7052, 0.3419, 0.0957, 0.4375, 0.9320, 0.1000, 0.1969, 0.9000])
MOLECULAR_MASS = np.array([89.094, 174.203, 132.119, 133.104, 121.154, 146.146, 147.131, 75.067, 155.156, 131.175,
                           131.175, 146.189, 149.208, 165.192, 115.132, 105.093, 119.119, 204.228, 181.191, 117.148])

PP_LIST = [PKA_AMINO_GROUP, PKA_CARBOXYL_GROUP, EIIP, LONE_ELECTRON_PAIRS, WIENER_INDEX, MOLECULAR_MASS]

DB_PATH = "~/swissprot/uniprot_sprot.fasta"

# Amino Acid Composition (AAC) groups - Polarity Charge
# C1; C2; C3; C4 
# (polar amino acid with positive charge, polar amino acid with negative charge, noncharged
# polar amino acid, nonpolar amino acid).

AAC_C1 = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'M', 'P']
AAC_C2 = ['S', 'T', 'C', 'Y', 'N', 'Q']
AAC_C3 = ['D', 'E']
AAC_C4 = ['R', 'K', 'H']

AAC_C_LIST = [AAC_C1, AAC_C2, AAC_C3, AAC_C4]

# Amino Acid Composition (AAC) groups - Hydrohpobicity
# H1;H2;H3;H4  (strong hydrophobic residue, weak hydrophobic residue, strong hydrophilic residue, weak hydrophilic residue).
# This scale is obtained from Kyte and Doolittle (1982). 
# K&D scale from 0 to +-2.0 is considered weak, >2.0 is strong hydrophobicity, and <-2.0 is strong hydrophilic. 


AAC_H1 = ['I', 'V', 'L', 'F', 'C']
AAC_H2 = ['M', 'A']
AAC_H3 = ['H', 'Q', 'N', 'E', 'D', 'K', 'R']
AAC_H4 = ['G', 'T', 'S', 'W', 'Y', 'P']

AAC_H_LIST = [AAC_H1, AAC_H2, AAC_H3, AAC_H4]


DNA_BINDING_GO_TERM_LIST = json.load(open("descendant_ids.json", "r"))

def get_instances_from_seq(seq : str, window_size : int = 9) -> list :
    instances = list()
    for i in range(len(seq) - window_size + 1):
        instances.append(seq[i:i+window_size])
    return instances
    
 
def rescale_pssm(input_pssm) -> np.ndarray:
    input_pssm = 1/(1 + np.exp(-input_pssm))
    return input_pssm

# Get only the pssm rows that are relevant to the sequence
def get_sliced_pssm(original_pssm : np.ndarray, start_index : int, window_size : int = 9):
    return original_pssm[start_index : start_index + window_size, :]

def create_pssm_pp(pssm_matrix : np.ndarray, pp_matrix : np.ndarray) -> np.ndarray:
    pssm_matrix = np.array(pssm_matrix, dtype=float)
    return pp_matrix @ pssm_matrix

def run_hmmscan(input_seq: str):
    """Run hmmscan against a given sequence."""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta') as temp_fasta:
        SeqIO.write([SeqRecord(Seq(input_seq), id='Query')], temp_fasta, 'fasta')
        temp_fasta.seek(0)
        output_file = 'r_d.out'
        subprocess.run(['hmmscan', '--domtblout', output_file, '--domE', '1e-05', 'pfam/Pfam-A.hmm', temp_fasta.name],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return output_file

def parse_hmmscan_output(domtblout: str):
    """Parse the hmmscan output to extract relevant data."""
    if not os.path.exists(domtblout):
        print("Output file not found.")
        return pd.DataFrame()

    with open(domtblout, 'r') as f:
        lines = [line for line in f if not line.startswith('#')]

    if not lines:
        print("No Pfam IDs found")
        return pd.DataFrame()

    df = pd.DataFrame(lines, columns=['line'])
    df = df['line'].str.split(expand=True)
    df = df.iloc[:, [1, 3, 5, 17, 18]].rename(columns={1: "pfam_id", 3: "query_id", 5: "qlen",  17: "start", 18: "end"})
    df['pfam_id'] = df['pfam_id'].str.split('.').str[0]
    df[['start', 'end']] = df[['start', 'end']].astype(int)
    return df

def add_go_terms(input_df : pd.DataFrame) -> pd.DataFrame:
    if input_df.empty:
        print("No Pfam IDs found, input is empty Dataframe")
        input_df['go_terms'] = None
        return input_df
    
    pfam_list = list([x for x in input_df.iloc[:, 0]])
    result = pfam2go(pfam_list)
    # print(result)
    
    if not pfam_list or result is None:
        print("No Pfam IDs found, input is empty Dataframe")
        input_df['go_terms'] = None
        return input_df

    # pfam_to_go = dict(zip(result["Pfam accession"], result["GO accession"]))
    result = result.groupby('Pfam accession')['GO accession'].agg(list).reset_index()
    pfam_to_go = dict(zip(result["Pfam accession"], result["GO accession"]))
    # print(pfam_to_go)
    
    input_df['go_terms'] = input_df.iloc[:, 0].apply(lambda x: pfam_to_go.get(x.split('.')[0], ''))
    # print(input_df)
    return input_df

def label_dna_binding(input_df : pd.DataFrame, seq_len : int) -> np.ndarray:
    label_array = np.zeros(seq_len) 
    
    if input_df.empty:
        print("No GO terms found")
        return label_array
    
    input_df['is_dna_binding'] = input_df['go_terms'].apply(
        lambda go_terms: 1 if go_terms is not None and any(go in DNA_BINDING_GO_TERM_LIST for go in go_terms) else 0
    )    
    for index, row in input_df.iterrows():
        if row['is_dna_binding'] == 1:
            label_array[row['start']:row['end'] + 1] = 1
    
    pd.display(input_df)
    
    return label_array
    
    
def extract_features_for_dataset(input_fasta_dataset : str):
    full_list = list()
    
    for record in SeqIO.parse(input_fasta_dataset, "fasta"):
        extract_individual_features(record)
    


def extract_individual_features(input_fasta : SeqRecord.SeqRecord):
    """Extract all features for a single sequence in FASTA input format."""    
    seq_list = get_instances_from_seq(input_fasta.seq)
    
    # =============================================================
    # Extract PSSM
    output_pssm = input_fasta.id + ".pssm"
    num_iterations = 3  

    subprocess.run(["psiblast", "-query", input_fasta, "-db", DB_PATH, 
                "-out_ascii_pssm", output_pssm, "-num_iterations", str(num_iterations), "-evalue", "0.001"], 
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    pssm_df = pd.read_csv(output_pssm, delim_whitespace=True, skiprows=3, header=None)
    os.remove(output_pssm)  
    pssm_array = pssm_df.iloc[:-5, 2:22].to_numpy(dtype=int)
    pssm_array = rescale_pssm(pssm_array)
    
    pssm_pp_array = list()
    
    for index, seq in enumerate(seq_list):
        
    
    os.remove(output_pssm)
    # =============================================================
    
    # =============================================================
    # Extract Domains
    domain_array = label_dna_binding(add_go_terms(parse_hmmscan_output(run_hmmscan(input_fasta.seq))))
    # =============================================================
    
    

    
    