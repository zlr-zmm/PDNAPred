import numpy as np
import torch,esm
import re,sys
from transformers import AutoTokenizer, AutoModel
torch.cuda.set_device (3)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
transformer_link = "esm2"
print("Loading: {}".format(transformer_link))
tokenizer = AutoTokenizer.from_pretrained(transformer_link)
model = model.to(device)
model = model.eval()

data_path = sys.argv[1]
read_path = "dataset/"+data_path+".txt"
save_path = "embeddings/ESM2_"+data_path+".csv"
def read_fa(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    i=0
    seq_list = []
    labels_list = []
    for line in lines:
      if (i==0):
        line = line.replace('\n','')
        seq_list.append(line)
        i=1
      elif (i==1):
        line = line.replace('\n', '')
        line = re.findall(".{1}",line)
        line = " ".join(line)
        labels_list.append(line)
        i = 0
    return seq_list, labels_list
seqs_all, labels_all = read_fa(read_path)


number=0
for seq in seqs_all:
    number = number+1
    print(number)
    print(seq)
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
    with open(save_path,'a') as f:
        np.savetxt(f, results["representations"][33][:, 1:len(seq) + 1].data.squeeze().cpu().numpy(), delimiter=',')
        torch.cuda.empty_cache()