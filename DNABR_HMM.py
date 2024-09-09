import numpy as np
import pandas as pd
import os
import subprocess
import tempfile
import pickle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix, recall_score, f1_score
from sklearn.utils import shuffle
import json
from pfam2go import pfam2go 
import argparse
warnings.filterwarnings("ignore")

AMINO_ACID_INDICES = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 
                      'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

DB_PATH = "databases/uniprot_sprot.fasta"

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

AAC_C1 = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'M', 'P']
AAC_C2 = ['S', 'T', 'C', 'Y', 'N', 'Q']
AAC_C3 = ['D', 'E']
AAC_C4 = ['R', 'K', 'H']

AAC_C_LIST = [AAC_C1, AAC_C2, AAC_C3, AAC_C4]

AAC_H1 = ['I', 'V', 'L', 'F', 'C']
AAC_H2 = ['M', 'A']
AAC_H3 = ['H', 'Q', 'N', 'E', 'D', 'K', 'R']
AAC_H4 = ['G', 'T', 'S', 'W', 'Y', 'P']

AAC_H_LIST = [AAC_H1, AAC_H2, AAC_H3, AAC_H4]

DNA_BINDING_GO_TERM_LIST = json.load(open("descendant_ids.json", "r"))

def create_pp_matrix() -> np.ndarray:
    pp_matrix = np.empty((len(PP_LIST), len(AMINO_ACID_INDICES)), dtype=float)
    for i, pp in enumerate(PP_LIST):
        max_val = np.max(pp)
        min_val = np.min(pp)
        pp_matrix[i] = (pp - min_val) / (max_val - min_val)
    
    return pp_matrix

# Constant PP_MATRIX
PP_MATRIX = create_pp_matrix()

obv_classes = {
    'A' : 0, 'G' : 0, 'V' : 0,
    'I': 1, 'L': 1, 'F': 1, 'P': 1,
    'Y': 2, 'M': 2, 'T': 2, 'S': 2,
    'H': 3, 'N': 3, 'Q': 3, 'W': 3,
    'R': 4, 'K': 4,
    'D': 5, 'E': 5,
    'C': 6
}

def generate_obv(amino_acid):
    temp = np.zeros(7)
    temp[obv_classes.get(amino_acid)] = 1
    return temp

def get_instances_from_seq(seq : str, window_size : int = 9) -> list :
    instances = list()
    for i in range(len(seq) - window_size + 1):
        instances.append(seq[i:i+window_size])
    return instances


"""Generate PSSM using psiblast from a given sequence."""
def generate_pssm(input_seq: str, num_iterations = 3) -> np.ndarray:
    DB_PATH = "./databases/uniprot_sprot.fasta"
    output_pssm = "output.pssm"

    # Creating a temporary fasta file for input
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as temp_fasta:
        SeqIO.write([SeqRecord(Seq(input_seq))], temp_fasta, "fasta")
        temp_fasta_path = temp_fasta.name

    # Running psiblast
    try:
        subprocess.run(["psiblast", "-query", temp_fasta_path, "-db", DB_PATH, 
                        "-out_ascii_pssm", output_pssm, "-num_iterations", str(num_iterations), "-evalue", "0.001"], 
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    finally:
        os.remove(temp_fasta_path)

    # Reading PSSM output
    try:
        pssm_df = pd.read_csv(output_pssm, delim_whitespace=True, skiprows=3, header=None)
        os.remove(output_pssm)  # Clean up PSSM file after reading
        pssm_array = pssm_df.iloc[:-5, 2:22].to_numpy(dtype=int)
        return pssm_array
    except FileNotFoundError:
        print(f"Error: PSSM file not found. Input Sequence: {input_seq}")
        return None
    

# Rescale pssm using sigmoid
def rescale_pssm(input_pssm) -> np.ndarray:
    input_pssm = 1/(1 + np.exp(-input_pssm))
    return input_pssm

# Get only the pssm rows that are relevant to the sequence
def get_sliced_pssm(original_pssm : np.ndarray, start_index : int, window_size : int = 9):
    return original_pssm[start_index : start_index + window_size, :]

def create_pssm_pp(pssm_matrix : np.ndarray, pp_matrix : np.ndarray) -> np.ndarray:
    pssm_matrix = np.array(pssm_matrix, dtype=float)
    return pp_matrix @ pssm_matrix

def calculate_AAC_PC(seq : str):
    window_size = len(seq)
    
    def get_c_i():
        c_i = np.zeros((4, window_size - 1), dtype=int)
        for gap in range(1, window_size):
            for j in range(window_size - gap):
                for index, aac_class in enumerate(AAC_C_LIST):
                    if seq[j] in aac_class and seq[j + gap] in aac_class:
                        c_i[index][gap - 1] += 1
        # print(c_i)
        return c_i
    
    def get_n_i():
        n_i = [np.sum(seq.count(a) for a in aac_class) for aac_class in AAC_C_LIST]
        # print(n_i)
        return np.array(n_i)
    
    c_i = get_c_i()
    n_i = get_n_i()
    
    output_aac_list = list()
    for i in range(0, 4):
        sum = 0
        for k in range(0, window_size - 1):
            first_term = ((c_i[i][k] / (window_size - k)) - (n_i[i]**2 / window_size**2))
            if np.isnan(first_term):
                first_term = 0
            second_term = np.square(first_term) / (2 * (n_i[i]**2 / window_size**2))
            if np.isnan(second_term):
                second_term = 0
            sum += (first_term + second_term)
        output_aac_list.append(sum)
    
    # print(output_aac_list)
    return output_aac_list

def calculate_AAC_H(seq : str):
    window_size = len(seq)
    def get_h_i():
        h_i = np.zeros((4, window_size - 1), dtype=int)
        for gap in range(1, window_size):
            for j in range(window_size - gap):
                for index, aac_class in enumerate(AAC_H_LIST):
                    if seq[j] in aac_class and seq[j + gap] in aac_class:
                        h_i[index][gap - 1] += 1
        # print(h_i)
        return h_i
    
    def get_m_i():
        m_i = [np.sum(seq.count(a) for a in aac_class) for aac_class in AAC_H_LIST]
        # print(m_i)
        return np.array(m_i)
    
    h_i = get_h_i()
    m_i = get_m_i()
    
    output_aac_list = list()
    for i in range(0, 4):
        sum = 0
        for k in range(0, window_size - 1):
            first_term = ((h_i[i][k] / (window_size - k)) - (m_i[i]**2 / window_size**2))
            first_term = 0 if np.isnan(first_term) else first_term
            second_term = np.square(first_term) / (2 * (m_i[i]**2 / window_size**2))
            second_term = 0 if np.isnan(second_term) else second_term
            sum += (first_term + second_term)
        output_aac_list.append(sum)
    
    # print(output_aac_list)
    return output_aac_list

def get_full_obv(seq: str):
    full_obv = np.zeros((len(seq), 7)) 
    for idx, aa in enumerate(seq):
        full_obv[idx] = generate_obv(aa)  
    return full_obv.flatten()

def pre_generate_pssm(input_df, file_name:str, window_size : int):
    pssm_list = list()
    for seq in input_df['seq']:
        pssm = generate_pssm(seq)
        pssm_list.append(pssm)
    
    with open(file_name, 'wb') as f:
        pickle.dump(pssm_list, f)
    return pssm_list
    
def gen_disorder_pred(input_seq: str) -> np.ndarray:
    # create temporary fasta file based on input seq
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as temp_fasta:
        SeqIO.write([SeqRecord(Seq(input_seq))], temp_fasta, "fasta")
        temp_fasta_path = temp_fasta.name
    
    # run IUPred, save stdout to a file
    with open("disorder_pred.txt", "w") as output_file:
        subprocess.run(["python3", "iupred3/iupred3.py", temp_fasta_path, "long"], check=True, stdout=output_file, stderr=subprocess.STDOUT)
    
    # parse output file and convert to a numpy array
    with open("disorder_pred.txt", "r") as output_file:
        lines = output_file.readlines()
        # Get line 13 onwards, and only the 3rd column
        disorder_pred = np.array([float(line.split()[2]) for line in lines[12:]])
        
    # clean up files
    os.remove(temp_fasta_path)
    return disorder_pred

def gen_disorder_pred_for_dataset(input_file):
    df = pd.read_csv(input_file)
    disorder_preds = list()
    for seq in df['seq']:
        disorder_pred = gen_disorder_pred(seq)
        disorder_preds.append(disorder_pred)
    
    # save to pickle file
    with open("disorder_preds.pkl", "wb") as f:
        pickle.dump(disorder_preds, f)
        
    return disorder_preds

def get_diso_pred_for_seq(seq_index: int):
    with open("disorder_preds.pkl", "rb") as f:
        disorder_preds = pickle.load(f)
    return disorder_preds[seq_index]

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
    
    # pd.display(input_df)
    
    return label_array

def process_sequences(input_csv, output_file):
    # The fasta file is a list of sequences 
    # Open the fasta file, processs them sequence by sequence
    list_of_domain_annotations = list()
    
    df = pd.read_csv(input_csv)
    for seq in df['seq']:
        run_hmmscan(seq)
        pfam_df = add_go_terms(parse_hmmscan_output("r_d.out"))
        res = label_dna_binding(pfam_df, len(seq))
        print(res)
        list_of_domain_annotations.append(res)
    
    # list_of_domain_annotations = np.array(list_of_domain_annotations)
    # np.save("dom_annotations_test.npy", list_of_domain_annotations)
    print(len(list_of_domain_annotations))
    pickle.dump(list_of_domain_annotations, open(output_file, "wb"))
    return list_of_domain_annotations

def get_domain_array_from_file(file_name):
    return np.load(file_name)


# Assuming these functions are correctly implemented
def get_all_features_for_one_sequence(full_seq: str, dna_label: str, input_pssm : np.ndarray, seq_diso_values, domain_values, input_hhm : np.ndarray, window_size: int = 9) -> list:
    seq_list = get_instances_from_seq(full_seq, window_size=window_size)  # Assuming this returns a list of sequences of length window_size
    pssm = rescale_pssm(input_pssm)  # Assuming this returns a PSSM for the full_seq
    # pssm = np.array(input_pssm)[:, :20]
    # print(pssm.shape)

    all_features_list = []  # Use a list to maintain structure

    for index, seq in enumerate(seq_list):
        # print(f"Processing sequence {index} of {len(seq_list)}")
        current_residue_label = dna_label[index + window_size // 2]
        if current_residue_label == '2':
            # print(f"Residue unknown at index {index}, skipping")
            continue
        
        pssm_pp_features = create_pssm_pp(get_sliced_pssm(pssm, index, window_size).T, PP_MATRIX).flatten()
        aac_features = np.append(calculate_AAC_PC(seq), calculate_AAC_H(seq))
        obv_features = get_full_obv(seq) 
        diso_features = [0 if x < 0.5 else 1 for x in seq_diso_values[index:index+window_size]]
        domain_features = domain_values[index:index+window_size]
        hhm_features = np.array(input_hhm[index:index+window_size]).flatten()
        
        all_features = np.concatenate([pssm_pp_features, aac_features, obv_features, domain_features])
        # all_features = np.concatenate([aac_features, obv_features, diso_features, domain_features])
        all_features_list.append((all_features, current_residue_label))

    return all_features_list

# Generate feature vectors for each sequence in the training dataset
def get_all_features_for_dataset(dataset: pd.DataFrame, generated_pssm_file, generated_diso_file, generated_domain_file, generated_hhm_file, window_size : int = 9) -> list:
    full_pssm = list(pickle.load(open(generated_pssm_file, 'rb'))) 
    full_diso_values = list(pickle.load(open(generated_diso_file, 'rb')))   
    full_domain_values = list(pickle.load(open(generated_domain_file, 'rb')))  
    full_hhm_values = list(pickle.load(open(generated_hhm_file, 'rb')))
    
    all_features_list = []
    for index, row in dataset.iterrows():
        if full_pssm[index] is None:
            print(f"Skipping sequence at index {index} due to missing PSSM")
            continue
        if full_diso_values[index] is None:
            print(f"Diso value not available! Index: {index}")
            continue
        # if full_domain_values[index] is None:
        #     print(f"Domain value not available! Index: {index}")
        #     continue
        try:
            all_features_list.extend(get_all_features_for_one_sequence(full_seq=row['seq'], 
                                                                       dna_label=row['dna_label'],
                                                                       input_pssm = full_pssm[index], 
                                                                       seq_diso_values=full_diso_values[index],
                                                                       domain_values=full_domain_values[index],
                                                                       input_hhm=full_hhm_values[index],
                                                                       window_size=window_size))
        except FileNotFoundError as e:
            print(f"Error processing sequence at index {index}: {e}")
            continue
    return all_features_list

# EXTRACT
def extract_features_for_dataset(input_fasta_dataset : str) -> list:    
    full_list = list()
    for record in SeqIO.parse(input_fasta_dataset, "fasta"):
        individual_feature = extract_individual_features(record)
        full_list.append((record.seq, individual_feature))
    return full_list
        
def extract_individual_features(input_fasta) -> np.ndarray:
    """Extract all features for a single sequence in FASTA input format."""    
    seq_list = get_instances_from_seq(input_fasta.seq)
    all_features_list = list()

    # =============================================================
    # Extract PSSM
    output_pssm = "output.pssm"
    num_iterations = 3  
    window_size = 9

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as temp_fasta:
        SeqIO.write([SeqRecord(input_fasta.seq)], temp_fasta, "fasta")
        temp_fasta_path = temp_fasta.name    

    subprocess.run(["psiblast", "-query", temp_fasta_path, "-db", DB_PATH, 
                "-out_ascii_pssm", output_pssm, "-num_iterations", str(num_iterations), "-evalue", "0.001"], 
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
    
    pssm_df = pd.read_csv(output_pssm, delim_whitespace=True, skiprows=3, header=None)
    os.remove(output_pssm)  
    pssm_array = pssm_df.iloc[:-5, 2:22].to_numpy(dtype=int)
    pssm = rescale_pssm(pssm_array)
    
    # =============================================================
    
    # =============================================================
    # Extract Domains
    domain_values = label_dna_binding(add_go_terms(parse_hmmscan_output(run_hmmscan(input_fasta.seq))), len(input_fasta.seq))
    # =============================================================
    
    for index, seq in enumerate(seq_list):
        pssm_pp_features = create_pssm_pp(get_sliced_pssm(pssm, index, window_size).T, PP_MATRIX).flatten()
        aac_features = np.append(calculate_AAC_PC(seq), calculate_AAC_H(seq))
        obv_features = get_full_obv(seq) 
        # diso_features = [0 if x < 0.5 else 1 for x in seq_diso_values[index:index+window_size]]
        domain_features = domain_values[index:index+window_size]
        # hhm_features = np.array(input_hhm[index:index+window_size]).flatten()
    
        all_features = np.concatenate([pssm_pp_features, aac_features, obv_features, domain_features])
        all_features_list.append(all_features)
        # print(all_features)
        np.array(all_features_list)
    return all_features_list

def rf_model_predict(input_features : np.ndarray) -> np.ndarray:
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    
    return rf_model.predict(input_features)

def write_tuple_to_file(filename, data):
    string, values = data
    
    # Ensure the length of string and values match
    # if len(string) != len(values):
    #     raise ValueError("The length of the string and the values list must be the same.")
    
    # Open the file in write mode
    with open(filename, 'w') as file:
        # Iterate over characters and their corresponding values
        for char, value in zip(string, values):
            file.write(f"{char} {value}\n")

def main():
    parser = argparse.ArgumentParser(description="Extract sequences from FASTA file and store them separately.")
    parser.add_argument('-i', '--input_file', type=str, required=True, help="Input FASTA file")
    parser.add_argument('-d', '--db', type=str, required=False, help="Output prefix for the extracted sequences")
    args = parser.parse_args()
    DB_PATH = args.db
    input_file = args.input_file
    
    extracted_features = extract_features_for_dataset(input_file)
    
    for sequence in extracted_features:
        features = sequence[1]
        output = rf_model_predict(features)
        output = np.pad(output, (4, 4), 'constant')
        print(output)
        write_tuple_to_file("output_1.txt", (sequence[0], output))

if __name__ == '__main__':
    main()

    