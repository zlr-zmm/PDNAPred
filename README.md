If you have any questions regarding the code/paper/data, please contact Lingrong Zhang via zlr_zmm@163.com

Repo for PDNAPred framework
This repo holds the code for PDNAPred feamework for protein-DNA binding sites prediction. We also provide protein-RNA binding sites prediction.
PDNA is primarily dependent on two large-scale pre-trained protein language model: ESM-2 and ProtT5 implemented using HuggingFace’s Transformers and Pytorch. Please install the dependencies in advance.
ESM-2: https://huggingface.co/facebook/esm2_t12_35M_UR50D
ProtT5: https://huggingface.co/Rostlab/prot_t5_xl_uniref50

Usage:
We provide the python script for predicting protein-DNA binding sites of given protein sequences in FASTA format. Here we provide a sample 

