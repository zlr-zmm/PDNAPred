from transformers import T5Tokenizer, T5EncoderModel, BertGenerationEncoder, BertTokenizer
import torch
import re
import numpy as np
import sys
torch.cuda.set_device (3)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))
transformer_link = "../prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))
model = T5EncoderModel.from_pretrained(transformer_link)
if device==torch.device("cuda"):
  model.to(torch.float32) # only cast to full-precision if no GPU is available
model = model.to(device)
model = model.eval()
tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, legacy=True )
data_path = sys.argv[1]
read_path = "../data/"+data_path+".txt"
save_path = "../embeddings/ProtT5_"+data_path+".csv"
save_label_path = "../embeddings/ProtT5_"+data_path+"_label.csv"
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
      else:
        line = line.replace('\n', '')
        line = re.findall(".{1}",line)
        line = " ".join(line)
        labels_list.append(line)
        i = 0
    return seq_list, labels_list
seqs_all, labels_all = read_fa(read_path)
print(seqs_all)

number=0
for seq in seqs_all:
    label = labels_all[number]
    label = np.fromstring(label, dtype=int, sep=" ")
    number = number+1
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) ]
    ids = tokenizer(sequence_examples, add_special_tokens=False, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    emb_0 = embedding_repr.last_hidden_state[0,:len(sequence_examples[0])]
    with open(save_path,'a') as f:
        np.savetxt(f, emb_0.data.cpu().numpy(),  delimiter=',')
        torch.cuda.empty_cache()
    with open(save_label_path,'a') as f:
        np.savetxt(f, label,  delimiter=',')
        torch.cuda.empty_cache()
