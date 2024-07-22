
# PDNAPred Framework

This repository contains the code for the PDNAPred framework, which is used for predicting protein-DNA binding sites. We also provide functionality for predicting protein-RNA binding sites.

PDNAPred relies on two large-scale pre-trained protein language models: ESM-2 and ProtT5. These models are implemented using Hugging Face's Transformers library and PyTorch. Please make sure to install the required dependencies beforehand.

- ESM-2: [https://huggingface.co/facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_35M_UR50D)
- ProtT5: https://huggingface.co/Rostlab/prot_t5_xl_uniref50

# Usage

To predict protein-DNA binding sites for given protein sequences in FASTA format, we provide a Python script. Here is an example:

# Contact

If you have any questions regarding the code, paper, or data, please feel free to contact Lingrong Zhang at [zlr_zmm@163.com](mailto:zlr_zmm@163.com).
