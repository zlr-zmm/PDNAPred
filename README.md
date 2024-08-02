# PDNAPred Framework

This repository contains the code for the PDNAPred framework, which is used for predicting protein-DNA binding sites. We also provide functionality for predicting protein-RNA binding sites.

PDNAPred relies on two large-scale pre-trained protein language models: ESM-2 and ProtT5. These models are implemented using Hugging Face's Transformers library and PyTorch. Please make sure to install the required dependencies beforehand.

- ESM-2: [https://huggingface.co/facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_35M_UR50D)
- ProtT5: https://huggingface.co/Rostlab/prot_t5_xl_uniref50

# Usage

First, you need to download the model weights for ESM-2 and ProtT5 from the provided Hugging Face URLs. Please visit the above links to download the respective weight files.

Save the downloaded weight files in your working directory and make sure you know their exact paths.

Next, you will use the provided `feature_ProtT5.py` and `feature_ESM2.py` scripts to generate embedding features for ProtT5 and ESM-2, respectively. In these scripts, you need to modify the file paths according to your needs.

For the `feature_ProtT5.py` and `feature_ESM2.py` scripts, you need to run them separately to generate embedding features for ProtT5 and ESM-2. Run the following commands:

```
python feature_ProtT5.py
python feature_ESM2.py
```

This will generate the corresponding embedding features files.

Finally, you can proceed with model training and validation using the provided `Train.py` script. Before running it, make sure you have prepared the training data and labels, and have downloaded the weight files and generated embedding feature files.

In the `Train.py` script, you need to modify the file paths and other parameters according to your needs. Run the following command to start the model training and validation:

```
python Train.py
```

The script will train the model and perform validation based on the data and parameters you provided, and it will save the output in the specified output directory.

Please note that the file paths and other parameters in the above steps need to be modified according to your own setup. Make sure you have installed the required dependencies properly, and follow the steps in the specified order.

# Contact

If you have any questions regarding the code, paper, or trained model, please feel free to contact Lingrong Zhang at [zlr_zmm@163.com](mailto:zlr_zmm@163.com).
