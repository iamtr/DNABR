#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import os
import subprocess
import tempfile
import multiprocessing
import sys
import pickle
from line_profiler import LineProfiler
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix, recall_score, f1_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.utils import shuffle
from scipy import stats
import shap
import matplotlib
warnings.filterwarnings("ignore")


# ### Required Physicochemical Properties
# 
# Most sources can be found on AAindex
# 
# PKA source: D.R. Lide, Handbook of Chemistry and Physics, 72nd Edition, CRC Press, Boca Raton, FL, 1991. (Sigma Aldrich website)
# 
# EIIP: Electron-ion interaction potential (Veljkovic et al., 1985)
# 
# LEP: No citation, sorta implicit (NOT VERIFIED!)
# 
# Wiener Index: ?
# 
# Molecular Mass: Wikipedia, implicit

# In[4]:


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

# pKa_amino_group = np.array([9.87, 8.99, 8.72, 9.90, 10.70, 9.13, 9.47, 9.78,
#                            9.33, 9.76, 9.74, 9.06, 9.28, 9.31, 10.64, 9.21, 9.10, 9.41, 9.21, 9.74])
# pKa_carboxyl_group = np.array([2.35, 1.82, 2.14, 1.99, 1.92, 2.17, 2.10, 2.35,
#                               1.80, 2.32, 2.33, 2.16, 2.13, 2.20, 1.95, 2.19, 2.09, 2.46, 2.20, 2.29])
# eiip = np.array([0.0373, 0.0959, 0.0036, 0.1263, 0.0829, 0.0761, 0.0057, 0.0050, 0.0242,
#                 0.0000, 0.0000, 0.0371, 0.0823, 0.0946, 0.0198, 0.0829, 0.0941, 0.0548, 0.0516, 0.0058])
# lone_electron_pairs = np.array(
#     [0, 0, 1, 2, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0])
# winer_index = np.array([0.3466, 0.1156, 0.3856, 0.2274, 0.0501, 0.6379, 0.1938, 0.1038, 0.2013,
#                        0.2863, 0.1071, 0.7767, 0.7052, 0.3419, 0.0957, 0.4375, 0.9320, 0.1000, 0.1969, 0.9000])
# molecular_mass = np.array([71.078, 156.186, 114.103, 115.087, 103.143, 128.129, 129.114, 57.051, 137.139,
#                           113.158, 113.158, 128.172, 131.196, 147.174, 97.115, 87.077, 101.104, 186.210, 163.173, 99.131])

PP_LIST = [PKA_AMINO_GROUP, PKA_CARBOXYL_GROUP, EIIP, LONE_ELECTRON_PAIRS, WIENER_INDEX, MOLECULAR_MASS]
# PP_LIST = [pKa_amino_group, pKa_carboxyl_group, eiip, lone_electron_pairs, winer_index, molecular_mass]


# In[5]:


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


# #### PP Matrix stored as a constant

# In[6]:


# rows: normalized pp properties 
# columns: amino acids
def create_pp_matrix() -> np.ndarray:
    pp_matrix = np.empty((len(PP_LIST), len(AMINO_ACID_INDICES)), dtype=float)
    for i, pp in enumerate(PP_LIST):
        max_val = np.max(pp)
        min_val = np.min(pp)
        pp_matrix[i] = (pp - min_val) / (max_val - min_val)
    
    return pp_matrix

# Constant PP_MATRIX
PP_MATRIX = create_pp_matrix()
# print(PP_MATRIX)


# ### OBV
# 
# Source: Shen, Juwen, et al. "Predicting protein–protein interactions based only on sequences information." Proceedings of the National Academy of Sciences 104.11 (2007): 4337-4341. (Supp. information)
# 
# Note: We use 7 classes here instead of 6. It was not mentioned why they used 6 classes only, when the source mentioned that amino acids are grouped into 7 classes

# In[7]:


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


# ### Get Window Instance from sequence

# In[8]:


# takes in a string, then 
# extract list of instances by sliding a window through the sequence
def get_instances_from_seq(seq : str, window_size : int = 9) -> list :
    instances = list()
    for i in range(len(seq) - window_size + 1):
        instances.append(seq[i:i+window_size])
    return instances
    


# ## Generate PSSM-PP

# In[9]:


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
                        "-out_ascii_pssm", output_pssm, "-num_iterations", str(num_iterations)], 
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    finally:
        os.remove(temp_fasta_path)

    # Reading PSSM output
    pssm_df = pd.read_csv(output_pssm, delim_whitespace=True, skiprows=3, header=None)
    os.remove(output_pssm)  # Clean up PSSM file after reading
    pssm_array = pssm_df.iloc[:-5, 2:22].to_numpy(dtype=int)
    return pssm_array

# Rescale pssm using sigmoid
def rescale_pssm(input_pssm) -> np.ndarray:
    input_pssm = 1/(1 + np.exp(-input_pssm))
    return input_pssm

# Get only the pssm rows that are relevant to the sequence
def get_sliced_pssm(original_pssm : np.ndarray, start_index : int, window_size : int = 9):
    return original_pssm[start_index : start_index + window_size, :]

# generate_pssm("KPKNKDKDKKVPEPDNKKKKPKKEEEQKWKWWEEERYPEGIKWKFLEHKGPVFAPPYEPLPENVKFYYDGKVMKLSPKAEEVATFFAKMLDHEYTTKEIFRKNFFKDWRKEMTNEEKNIITNLSKCDFTQMSQYFKAQTEARKQMSKEEKLKIKEENEKLLKEYGFCIMDNHKERIANFKIEPPGLFRGRGNHPKMGMLKRRIMPEDIIINCSKDAKVPSPPPGHKWKEVRHDNKVTWLVSWTENIQGSIKYIMLNPSSRIKGEKDWQKYETARRLKKCVDKIRNQYREDWKSKEMKVRQRAVALYFIDKLALRAGNEKEEGETADTVGCCSLRVEHINLHPELDGQEYVVEFDFLGKDSIRYYNKVPVEKRVFKNLQLFMENKQPEDDLFDRLNTGILNKHLQDLMEGLTAKVFRTYNASITLQQQLKELTAPDENIPAKILSYNRANRAVAILCNHQRAPPKTFEKSMMNLQTKIDAKKEQLADARRDLKSAKADAKVMKDAKTKKVVESKKKAVQRLEEQLMKLEVQATDREENKQIALGTSKLNFLDPRITVAWCKKWGVPIEKIYNKTQREKFAWAIDMADEDYE")


# In[10]:


def create_pssm_pp(pssm_matrix : np.ndarray, pp_matrix : np.ndarray) -> np.ndarray:
    return np.sqrt(pp_matrix) @ np.sqrt(pssm_matrix)


# ### Amino Acid Correlation

# In[11]:


# AAC_PC takes in a sequence of 9 amino acids then outputs a list of 4 values
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


# ### OBV

# In[12]:


def get_full_obv(seq: str):
    full_obv = np.zeros((len(seq), 7)) 
    for idx, aa in enumerate(seq):
        full_obv[idx] = generate_obv(aa)  
    return full_obv.flatten()


# ### Pre-generate PSSMs and store to a numpy file

# In[13]:


# Generate pssms for a list of sequences, then save them to a pickle file for future use
def pre_generate_pssm(input_df, file_name:str):
    pssm_list = list()
    for seq in input_df['seq']:
        pssm = generate_pssm(seq)
        pssm_list.append(pssm)
    
    with open(file_name, 'wb') as f:
        pickle.dump(pssm_list, f)
    return pssm_list
    
# list_of_train_pssms = pre_generate_pssm(pd.read_csv("./DRNA_TRAIN.csv"), "generated_pssms_train.pkl")
# print(len(list_of_train_pssms))
# list_of_test_pssms = pre_generate_pssm(pd.read_csv("./DRNA_TEST.csv"), "generated_pssms_test.pkl")
# print(len(list_of_test_pssms))


# ### Concatenate all features

# In[14]:


# Assuming these functions are correctly implemented
def get_all_features_for_one_sequence(full_seq: str, dna_label: str, input_pssm : np.ndarray, window_size: int = 9) -> list:
    seq_list = get_instances_from_seq(full_seq)  # Assuming this returns a list of sequences of length window_size
    pssm = rescale_pssm(input_pssm)  # Assuming this returns a PSSM for the full_seq

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
        all_features = np.concatenate([pssm_pp_features, aac_features, obv_features])
        # all_features = np.concatenate([obv_features])

        
        # # Ensure all_features is a 125D vector
        # if all_features.shape[0] != 125:
        #     raise ValueError(f"Feature vector for sequence '{seq}' is not 125D, but {all_features.shape[0]}D")
        all_features_list.append((all_features, current_residue_label))

    return all_features_list

# Generate feature vectors for each sequence in the training dataset
def get_all_features_for_dataset(dataset: pd.DataFrame, generated_pssm_file) -> list:
    full_pssm = list(pickle.load(open(generated_pssm_file, 'rb')))    
    
    all_features_list = []
    for index, row in dataset.iterrows():
        try:
            all_features_list.extend(get_all_features_for_one_sequence(full_seq=row['seq'], dna_label=row['dna_label'],
                                                                       input_pssm = full_pssm[index]))
        except FileNotFoundError as e:
            print(f"Error processing sequence at index {index}: {e}")
            continue
    return all_features_list

# %load_ext line_profiler
# %lprun -f get_all_features_for_one_sequence get_all_features_for_one_sequence("MKIAIINMGNNVINFKTVPSSETIYLFKVISEMGLNVDIISLKNGVYTKSFDEVDVNDYDRLIVVNSSINFFGGKPNLAILSAQKFMAKYKSKIYYLFTDIRLPFSQSWPNVKNRPWAYLYTEEELLIKSPIKVISQGINLDIAKAAHKKVDNVIEFEYFPIEQYKIHMNDFQLSKPTKKTLDVIYGGSFRSGQRESKMVEFLFDTGLNIEFFGNAREKQFKNPKYPWTKAPVFTGKIPMNMVSEKNSQAIAALIIGDKNYNDNFITLRVWETMASDAVMLIDEEFDTKHRIINDARFYVNNRAELIDRVNELKHSDVLRKEMLSIQHDILNKTRAKKAEWQDAFKKAID","000000000000110111000100000000000000000000000000000000000000000000010111111100000000000000000000000001000000000000111010000000000000000000000000000011000000000000000000000000000010000000001001111111000000000000000010101100001000000000011000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")


# In[15]:


# Assuming training_dataset is loaded correctly
training_dataset = pd.read_csv("DRNA_TRAIN.csv")
test_dataset = pd.read_csv("DRNA_TEST.csv")

all_training_features = get_all_features_for_dataset(training_dataset, "generated_pssms_train.pkl")
all_test_features = get_all_features_for_dataset(test_dataset, "generated_pssms_test.pkl")    

# Separate into X_train and y_train
X_train = [features for features, label in all_training_features]
y_train = [label for features, label in all_training_features]
X_test = [features for features, label in all_test_features]
y_test = [label for features, label in all_test_features]

# Optionally convert to numpy arrays for compatibility with scikit-learn
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# # **Compilation of results**

# ## No balancing

# In[23]:


def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: purple' if v else '' for v in is_max]

# Train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Get prediction probabilities
predictions = rf_model.predict_proba(X_test)
results = []

# Iterate over thresholds from 0.1 to 0.96 with a step of 0.02
for threshold in np.arange(0.1, 0.96, 0.02):
    y_pred = ['1' if p[1] >= threshold else '0' for p in predictions]
    
    mcc = matthews_corrcoef(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['0', '1']).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    results.append((threshold, mcc, sensitivity, specificity))


results_df = pd.DataFrame(results, columns=['Threshold', 'MCC', 'Sensitivity', 'Specificity'])
results_df = results_df.round(2)
styled_results_df = results_df.style.format("{:.2f}").apply(highlight_max, subset=['MCC'])
display(styled_results_df)


# ## Random Undersampling

# In[24]:


# Subsample X_train and y_train such that they contain equal amounts of positive and negative samples
# Assuming y_train contains binary labels where 1 is positive and 0 is negative
positive_indices = np.where(y_train == '1')[0]
negative_indices = np.where(y_train == '0')[0]


# # Determine the number of samples to subsample based on the smaller class
n_samples = min(positive_indices.shape[0], negative_indices.shape[0]) 

# Randomly select n_samples from both positive and negative indices
positive_subsample_indices = np.random.choice(positive_indices, n_samples, replace=False)
negative_subsample_indices = np.random.choice(negative_indices, n_samples, replace=False)
# unknkown_subsample_indices = np.random.choice(unknown_indices, n_samples, replace=True)

# Concatenate the subsampled indices and then use them to create subsampled X_train and y_train
subsample_indices = np.concatenate([positive_subsample_indices, negative_subsample_indices])
X_train_subsampled = X_train[subsample_indices]
y_train_subsampled = y_train[subsample_indices]
print(X_train_subsampled.shape, y_train_subsampled.shape)

shuffle_indices = np.random.permutation(len(X_train_subsampled))
X_train_subsampled = X_train_subsampled[shuffle_indices]
y_train_subsampled = y_train_subsampled[shuffle_indices]


# In[25]:


# Train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_subsampled, y_train_subsampled)

# Get prediction probabilities
predictions = rf_model.predict_proba(X_test)
results = []

# Iterate over thresholds from 0.1 to 0.96 with a step of 0.02
for threshold in np.arange(0.1, 0.96, 0.02):
    y_pred = ['1' if p[1] >= threshold else '0' for p in predictions]
    
    mcc = matthews_corrcoef(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['0', '1']).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    results.append((threshold, mcc, sensitivity, specificity))


results_df = pd.DataFrame(results, columns=['Threshold', 'MCC', 'Sensitivity', 'Specificity'])
results_df = results_df.round(2)
styled_results_df = results_df.style.format("{:.2f}").apply(highlight_max, subset=['MCC'])
display(styled_results_df)


# ## Random Oversampling

# In[26]:


positive_indices = np.where(y_train == '1')[0]
negative_indices = np.where(y_train == '0')[0]

n_samples = max(positive_indices.shape[0], negative_indices.shape[0]) 

positive_oversample_indices = np.random.choice(positive_indices, n_samples, replace=True)
negative_oversample_indices = np.random.choice(negative_indices, n_samples, replace=False)

oversample_indices = np.concatenate([positive_oversample_indices, negative_oversample_indices])
X_train_oversampled = X_train[oversample_indices]
y_train_oversampled = y_train[oversample_indices]
print(X_train_oversampled.shape, y_train_oversampled.shape)

shuffle_indices = np.random.permutation(len(X_train_oversampled))
X_train_oversampled = X_train_oversampled[shuffle_indices]
y_train_oversampled = y_train_oversampled[shuffle_indices]


# In[28]:


# Train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_oversampled, y_train_oversampled)

# Get prediction probabilities
predictions = rf_model.predict_proba(X_test)
results = []

# Iterate over thresholds from 0.1 to 0.96 with a step of 0.02
for threshold in np.arange(0.1, 0.96, 0.02):
    y_pred = ['1' if p[1] >= threshold else '0' for p in predictions]
    
    mcc = matthews_corrcoef(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['0', '1']).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    results.append((threshold, mcc, sensitivity, specificity))


results_df = pd.DataFrame(results, columns=['Threshold', 'MCC', 'Sensitivity', 'Specificity'])
results_df = results_df.round(2)
styled_results_df = results_df.style.format("{:.2f}").apply(highlight_max, subset=['MCC'])
display(styled_results_df)


# ## K-Means Ensemble Undersampling (20000 negative samples)

# In[29]:


n_clusters = 20000
iterations = 5

positive_indices = np.where(y_train == '1')[0]
negative_indices = np.where(y_train == '0')[0]
positive_features = X_train[positive_indices]
negative_features = X_train[negative_indices]

negative_cluster_centers = np.vstack([
    MiniBatchKMeans(n_clusters=n_clusters // iterations, random_state=i).fit(negative_features).cluster_centers_
    for i in range(iterations)
])

negative_labels = np.zeros(len(negative_cluster_centers))
positive_labels = np.ones(len(positive_features))
X_combined = np.vstack((negative_cluster_centers, positive_features))
y_combined = np.concatenate((negative_labels, positive_labels))
X_train_subsampled, y_train_subsampled = shuffle(X_combined, y_combined, random_state=42)

print(X_train_subsampled.shape, y_train_subsampled.shape)


# In[30]:


# Train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_subsampled, y_train_subsampled)

# Get prediction probabilities
predictions = rf_model.predict_proba(X_test)
results = []

# Iterate over thresholds from 0.1 to 0.96 with a step of 0.02
for threshold in np.arange(0.1, 0.96, 0.02):
    y_pred = ['1' if p[1] >= threshold else '0' for p in predictions]
    
    mcc = matthews_corrcoef(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['0', '1']).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    results.append((threshold, mcc, sensitivity, specificity))


results_df = pd.DataFrame(results, columns=['Threshold', 'MCC', 'Sensitivity', 'Specificity'])
results_df = results_df.round(2)
styled_results_df = results_df.style.format("{:.2f}").apply(highlight_max, subset=['MCC'])
display(styled_results_df)


# ## K-Means Ensemble Undersampling (60000 samples)

# In[33]:


n_clusters = 60000
iterations = 5

# negative_cluster_centers = np.vstack([
#     MiniBatchKMeans(n_clusters=n_clusters // iterations, random_state=i).fit(negative_features).cluster_centers_
#     for i in range(iterations)
# ])

negative_labels = np.zeros(len(negative_cluster_centers))
positive_labels = np.ones(len(positive_features))
X_combined = np.vstack((negative_cluster_centers, positive_features))
y_combined = np.concatenate((negative_labels, positive_labels))
X_train_subsampled, y_train_subsampled = shuffle(X_combined, y_combined, random_state=42)

print(X_train_subsampled.shape, y_train_subsampled.shape)


# In[34]:


# Train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_subsampled, y_train_subsampled)

# Get prediction probabilities
predictions = rf_model.predict_proba(X_test)
results = []

# Iterate over thresholds from 0.1 to 0.96 with a step of 0.02
for threshold in np.arange(0.1, 0.96, 0.02):
    y_pred = ['1' if p[1] >= threshold else '0' for p in predictions]
    
    mcc = matthews_corrcoef(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['0', '1']).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    results.append((threshold, mcc, sensitivity, specificity))


results_df = pd.DataFrame(results, columns=['Threshold', 'MCC', 'Sensitivity', 'Specificity'])
results_df = results_df.round(2)
styled_results_df = results_df.style.format("{:.2f}").apply(highlight_max, subset=['MCC'])
display(styled_results_df)

