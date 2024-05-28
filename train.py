import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, precision_score, f1_score, roc_auc_score, \
    roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.set_device(5)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(2304, 256, kernel_size=3, padding=1)  # 3x3的一维卷积层
        self.conv2 = nn.Conv1d(2304, 256, kernel_size=5, padding=2)  # 5x5的一维卷积层
        self.conv3 = nn.Conv1d(2304, 256, kernel_size=7, padding=3)  # 7x7的一维卷积层
        self.bn = nn.BatchNorm1d(256)
        self.act = nn.GELU()
        self.bigrucell = nn.GRU(768, 128, bidirectional=True)  # 双向GRU
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将通道维度放到最后
        x1 = self.conv1(x)
        x1 = self.bn(x1)
        x1 = self.act(x1)
        x1 = x1.permute(0, 2, 1)
        x2 = self.conv2(x)
        x2 = self.bn(x2)
        x2 = self.act(x2)
        x2 = x2.permute(0, 2, 1)
        x3 = self.conv3(x)
        x3 = self.bn(x3)
        x3 = self.act(x3)
        x3 = x3.permute(0, 2, 1)
        x = torch.cat((x1, x2, x3), dim=2)
        x, _ = self.bigrucell(x)  # 双向GRU
        x = self.mlp(x)  # MLP分类器
        x = torch.squeeze(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, features1, labels):
        self.features1 = torch.tensor(features1.values.astype(np.float32))
        self.labels = torch.tensor(labels.values.astype(np.float32))

    def __len__(self):
        return len(self.features1)

    def __getitem__(self, index):
        x1 = self.features1[index]
        y = self.labels[index]
        return x1, y

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

train_ProtT5_data1 = pd.read_csv("embeddings/ProtT5_DNA646Train.csv")
train_ESM2_data1 = pd.read_csv("embeddings/ESM2_DNA646Train.csv")
train_label = pd.read_csv("embeddings/ProtT5_DNA646Train_label.csv")

test_ProtT5_data1 = pd.read_csv("embeddings/ProtT5_DNA46Test.csv")
test_ESM2_data1 = pd.read_csv("embeddings/ESM2_DNA46Test.csv")
tes_label = pd.read_csv("embeddings/ProtT5_DNA46Test_label.csv")

x_train_ESM2 = train_ESM2_data1.iloc[:, 0:]
x_train_ProtT5 = train_ProtT5_data1.iloc[:, 0:]
x_train_label = train_label.iloc[:, 0]
x_test_ESM2 = test_ESM2_data1.iloc[:, 0:]
x_test_ProtT5 = test_ProtT5_data1.iloc[:, 0:]
x_test_label = tes_label.iloc[:, 0]
x_train = pd.concat([x_train_ESM2, x_train_ProtT5], axis=1)
x_test = pd.concat([x_test_ESM2, x_test_ProtT5], axis=1)
num_epochs = 20
batch_size = 32
num_folds = 5
# 7. 定义交叉验证
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
best_cnn_mcc = 0
best_transformer_mcc = 0
test_dataset = CustomDataset(x_test, x_test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
tprs = []
fprs = []
precisions = []
recalls = []
best_acc = 0
best_acc_model = None
mean_fpr_linspace = np.linspace(0, 1, 100)
best_model = None
all_spe = []
all_rec = []
all_pre = []
all_f1 = []
all_mcc = []
all_auc_final = []
all_acupr = []

print("Spe Rec Pre F1 MCC AUC")
for fold, (train_index, val_index) in enumerate(kf.split(x_train)):
    # print(f"Fold: {fold+1}")
    model_label = []
    train_data = x_train.iloc[train_index]
    train_labels = x_train_label.iloc[train_index]

    val_data = x_train.iloc[val_index]
    val_labels = x_train_label.iloc[val_index]

    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    model = Model()
    model.to(device)
    criterion = FocalLoss(alpha=0.1, gamma=2, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    for epoch in range(num_epochs):
        model.train()
        for data1, labels in train_loader:
            data1 = data1.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            final_output = model(data1.unsqueeze(1))
            scores = final_output
            model_label.append(labels.tolist())
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

        all_predictions = []
        all_labels = []
        all_auc = []
        model.eval()
        for data1, labels in val_loader:
            data1 = data1.to(device)
            final_output = model(data1.unsqueeze(1))
            scores = final_output.tolist()
            all_auc.extend(scores)
            final_output = (final_output.data > 0.5).int()
            model_label.append(labels.tolist())
            all_labels.extend(labels.tolist())
            all_predictions.extend(final_output.tolist())

        cm = confusion_matrix(all_labels, all_predictions)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(all_labels, all_predictions)
        MCC = matthews_corrcoef(all_labels, all_predictions)

        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_precision = precision_score(all_labels, all_predictions)
        val_roc = roc_auc_score(all_labels, all_auc)
        val_recall = recall_score(all_labels, all_predictions)
        val_f1 = f1_score(all_labels, all_predictions)
        precision, recall, _ = precision_recall_curve(all_labels, all_auc)
        fpr, tpr, _ = roc_curve(all_labels, all_auc)
        val_pr_auc = auc(recall, precision)
        num_samples = 100
        precision_sampled = np.linspace(0, 1, num_samples)
        recall_sampled = np.interp(precision_sampled, precision, recall)
        fpr_sampled = np.linspace(0, 1, num_samples)
        tpr_sampled = np.interp(fpr_sampled, fpr, tpr)
        fprs.append(fpr_sampled)
        tprs.append(tpr_sampled)
        precisions.append(precision_sampled)
        recalls.append(recall_sampled)
        # 最佳模型选择
        if MCC > best_acc:
            best_acc = MCC
            best_acc_model = model.state_dict().copy()
        all_spe.append(specificity)
        all_rec.append(val_recall)
        all_pre.append(val_precision)
        all_f1.append(val_f1)
        all_mcc.append(MCC)
        all_auc_final.append(val_roc)
        all_acupr.append(val_pr_auc)

print(f"{np.mean(all_spe):.4f}  {np.mean(all_rec):.4f}  {np.mean(all_pre):.4f}  {np.mean(all_f1):.4f}  "
      f"{np.mean(all_mcc):.4f}  {np.mean(all_auc_final):.4f}  {np.mean(all_acupr):.4f}")
mean_precision = np.mean(precisions, axis=0)
mean_recall = np.mean(recalls, axis=0)
mean_fpr = np.mean(fprs, axis=0)
mean_tpr = np.mean(tprs, axis=0)


val_pr_curve_data = pd.DataFrame({'Precision': mean_precision, 'Recall': mean_recall})
val_pr_curve_data.to_csv('save/PR/ESM2_ProtT5_DNA646_without_Att_5CV.csv', index=False)

val_roc_curve_data = pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr})
val_roc_curve_data.to_csv('save/ROC/ESM2_ProtT5_DNA646_without_Att_5CV.csv', index=False)
torch.save(best_acc_model, "save/model/ESM2_DNA646_without_Att.pt")
print("Spe Rec Pre F1 MCC AUC")

print("##### test #####")
best_model = Model()
best_model.eval()
best_model.to(device)
best_model.load_state_dict(torch.load("save/model/ESM2_DNA646_without_Att.pt"))

with torch.no_grad():
    all_predictions = []
    all_labels = []
    all_auc = []

    for data1,labels in test_loader:
        data1 = data1.to(device)
        final_output = best_model(data1.unsqueeze(1))
        scores = final_output.tolist()
        all_auc.extend(scores)
        final_output = (final_output.data > 0.5).int()
        all_labels.extend(labels.tolist())
        all_predictions.extend(final_output.tolist())

    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    MCC = matthews_corrcoef(all_labels, all_predictions)
    test_accuracy = accuracy_score(all_labels, all_predictions)
    test_precision = precision_score(all_labels, all_predictions)
    test_auc_roc = roc_auc_score(all_labels, all_auc)
    test_recall = recall_score(all_labels, all_predictions)
    test_f1 = f1_score(all_labels, all_predictions)
    precision, recall, _ = precision_recall_curve(all_labels, all_auc)
    fpr, tpr, _ = roc_curve(all_labels, all_auc)
    test_pr_auc = auc(recall, precision)
    test_roc_curve_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    test_roc_curve_data.to_csv('save/ROC/ESM2_ProtT5_DNA646_without_Att.csv', index=False)
    # Save the PR curve data for the test set to a CSV file
    test_pr_curve_data = pd.DataFrame({'Precision': precision, 'Recall': recall})
    test_pr_curve_data.to_csv('save/PR/ESM2_ProtT5_DNA646_without_Att.csv', index=False)
    print(f"{specificity:.4f} {test_recall:.4f} {test_precision:.4f} {test_f1:.4f} {MCC:.4f} {test_auc_roc:.4f}  {test_pr_auc:.4f}")
