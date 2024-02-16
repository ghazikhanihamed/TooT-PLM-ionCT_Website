NUM_CLASSES = 2
BATCH_SIZE = 1
NUM_EPOCHS = 100
DATASET_PATH = "./dataset/"
PLOT_PATH = "./plots/"
RESULTS_PATH = "./results/"
# SEQUENCE_IDS_DICT_PATH = "./dataset/sequence_ids_dict.jsn"
SEQUENCE_IDS_DICT_PATH = "./dataset/sequence_ids_dict_new.jsn"
ALL_SEQUENCES_PATH = "./dataset/all_sequences_new.csv"
FROZEN_REPRESENTATIONS_PATH = "./plm_representations/frozen_representations"
FINETUNED_REPRESENTATIONS_PATH = "./plm_representations/finetuned_representations"
PLM_REPRESENTATIONS_PATH = "./plm_representations/"
FINETUNED_RESULTS_PATH = "./finetuned_results"
FINETUNED_MODELS_PATH = "./finetuned_models/"
REPRESENTATIONS_PATH = "./representations/"
REPRESENTATIONS_FILTERED_PATH = REPRESENTATIONS_PATH + "representations_filtered/"
LATEX_PATH = "./latex/"
FROZEN = "frozen"
FINETUNED = "finetuned"
MAX_LENGTH_FINETUNED = 1024
MAX_LENGTH_FROZEN = 4096
SEED = 42
TRAIN = "train"
VAL = "val"
TEST = "test"
IONCHANNELS = "ionchannels"
IONTRANSPORTERS = "iontransporters"
MEMBRANE_PROTEINS = "membrane_proteins"
IONCHANNELS_MEMBRANEPROTEINS = "ionchannels_membraneproteins"
IONCHANNELS_MEMBRANEPROTEINS_BALANCED = "ionchannels_membraneproteins_balanced"
IONCHANNELS_MEMBRANEPROTEINS_IMBALANCED = "ionchannels_membraneproteins_imbalanced"
IONCHANNELS_IONTRANSPORTERS = "ionchannels_iontransporters"
IONTRANSPORTERS_MEMBRANEPROTEINS = "iontransporters_membraneproteins"
IONTRANSPORTERS_MEMBRANEPROTEINS_BALANCED = "iontransporters_membraneproteins_balanced"
IONTRANSPORTERS_MEMBRANEPROTEINS_IMBALANCED = (
    "iontransporters_membraneproteins_imbalanced"
)
TASKS = [
    "ionchannels_membraneproteins",
    "iontransporters_membraneproteins",
    "ionchannels_iontransporters",
]
PROTBERT = {"name": "ProtBERT", "model": "Rostlab/prot_bert"}
PROTBERTBFD = {"name": "ProtBERT-BFD", "model": "Rostlab/prot_bert_bfd"}
PROTT5 = {"name": "ProtT5", "model": "Rostlab/prot_t5_xl_half_uniref50-enc"}
ESM1B = {"name": "ESM-1b", "model": "facebook/esm1b_t33_650M_UR50S"}
ESM2 = {"name": "ESM-2", "model": "facebook/esm2_t33_650M_UR50D"}
ESM2_15B = {"name": "ESM-2_15B", "model": "facebook/esm2_t48_15B_UR50D"}
REPRESENTATIONS = [PROTBERT, PROTBERTBFD, PROTT5, ESM1B, ESM2, ESM2_15B]
# IC-MP: Ion Channels vs. Membrane Proteins/ IT-MP: Ion Transporters vs. Membrane Proteins/ IC-IT: Ion Channels vs. Ion Transporters
TASKS_SHORT = {
    "ionchannels_membraneproteins": "IC-MP",
    "iontransporters_membraneproteins": "IT-MP",
    "ionchannels_iontransporters": "IC-IT",
}
PLM_PARAM_SIZE = {
    "ProtBERT": "ProtBERT(420M)",
    "ProtBERT-BFD": "ProtBERT-BFD(420M)",
    "ProtT5": "ProtT5(3B)",
    "ESM-1b": "ESM-1b(650M)",
    "ESM-2": "ESM-2(650M)",
    "ESM-2_15B": "ESM-2(15B)",
}
PLM_ORDER = [
    "ProtBERT(420M)",
    "ProtBERT-BFD(420M)",
    "ESM-1b(650M)",
    "ESM-2(650M)",
    "ProtT5(3B)",
    "ESM-2(15B)",
]
PLM_ORDER_SHORT = ["ProtBERT", "ProtBERT-BFD", "ESM-1b", "ESM-2", "ProtT5", "ESM-2_15B"]
PLM_ORDER_FINETUNED = [
    "ProtBERT(420M)",
    "ProtBERT-BFD(420M)",
    "ESM-1b(650M)",
    "ESM-2(650M)",
]
PLM_ORDER_FINETUNED_SHORT = ["ProtBERT", "ProtBERT-BFD", "ESM-1b", "ESM-2"]
CLASSIFIER_ORDER = ["LR", "kNN", "SVM", "RF", "FFNN", "CNN"]

IC_MP_Train_DATASET = "IC_MP_train.csv"
IC_MP_Test_DATASET = "IC_MP_test.csv"
IT_MP_Train_DATASET = "IT_MP_train.csv"
IT_MP_Test_DATASET = "IT_MP_test.csv"
IC_IT_Train_DATASET = "IC_IT_train.csv"
IC_IT_Test_DATASET = "IC_IT_test.csv"

FINAL_MODELS_PATH = "./final_models/"
