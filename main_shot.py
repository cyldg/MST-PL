import os
import clip
import open_clip
import torch
import numpy as np
import torchvision
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import random
import omegaconf
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import Dataset
import helpers.data_helpers as dh
import methods.clip_transformations as CLIPTransformations
from utils import read_unknowns, nest_dict
from clip_utils import get_features, evaluate, zeroshot_classifier, get_ensamble_preds, get_pred_overlap, get_nn_metrics, load_embedding,load_aug_embedding
import methods.augmentations
from torchvision.datasets import ImageFolder
from prompt_CLIP import CustomCLIP_server, CustomCLIP_client,TextEncoder
from FASTPT import FaSTPT_Client,train_global,train_domain,train_domclassifier,Aggregation,test_infer
import torch.nn as nn

from iopath.common.file_io import PathManager as PathManagerBase
PathManager = PathManagerBase()
from datetime import datetime
mean=[0.5, 0.5, 0.5]
std = [0.5,0.5,0.5] 
class FEADATASET(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, groups: np.ndarray, domains: np.ndarray, filenames: np.ndarray):
        # 进行基本的数据一致性检查
        num_samples = features.shape[0]
        if not (labels.shape[0] == num_samples and
                groups.shape[0] == num_samples and
                domains.shape[0] == num_samples and
                filenames.shape[0] == num_samples):
            raise ValueError("所有输入数组的第一维度 (样本数量) 必须相同。")

        self.features = features
        self.labels = labels
        self.groups = groups  # 存储以便需要时访问
        self.domains = domains
        self.filenames = filenames # 存储以便需要时访问

    def __len__(self) -> int:

        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        # 假设标签是用于分类的整数类别，转换为 LongTensor
        # 如果标签是用于回归的浮点数，应使用 dtype=torch.float32
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # 假设领域是整数类别，转换为 LongTensor
        domain = torch.tensor(self.domains[idx], dtype=torch.long)

        return feature, label, domain


transform_train = transforms.Compose([
                transforms.Resize([256, 256]),            
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
        ])

transform_test = transforms.Compose([
                transforms.Resize([256, 256]),   
                transforms.CenterCrop([224,224]),         
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
        ])

parser = argparse.ArgumentParser(description='CLIP Advice')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
# flags = parser.parse_args()
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
args      = OmegaConf.merge(base, cfg, overrides)
if len(unknown) > 0:
    print(unknown)
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
args.yaml = flags.config

assert args.EXP.ADVICE_METHOD != 'CNN', "main.py not for CNN baseline, use train.py"


def flatten_config(dic, running_key=None, flattened_dict={}):
    for key, value in dic.items():
        if running_key is None:
            running_key_temp = key
        else:
            running_key_temp = '{}.{}'.format(running_key, key)
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            flatten_config(value, running_key_temp)
        else:
            #print(running_key_temp, value)
            flattened_dict[running_key_temp] = value
    return flattened_dict


torch.manual_seed(args.EXP.SEED)
np.random.seed(args.EXP.SEED)
random.seed(args.EXP.SEED)

output_folder = os.path.join(datetime.now().strftime("%m%d%Y_%H%M%S"))
PathManager.mkdirs(output_folder)
# logout = os.path.join(output_folder,args.EXP.LOGOUT)
logout = os.path.join(output_folder, f"{args.EXP.LOGOUT}_{args.EXP.SEED}_{args.EXP.N_CTX_GLOBAL}_{args.EXP.N_CTX_DOMAIN}_{args.EXP.SHOT}.txt")
DATASET_NAME = args.DATA.DATASET

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load(args.EXP.CLIP_MODEL, device)

# # load data
print("target domain:", args.DATA.TARGET)
cache_file = f"{args.DATA.SAVE_PATH}/{args.DATA.TARGET}/{args.EXP.IMAGE_FEATURES}_{args.EXP.CLIP_PRETRAINED_DATASET}_{args.EXP.CLIP_MODEL.replace('/','_')}.pt"


print(f"Loading test cached embeddings from {cache_file}")
test_features, test_labels, test_groups, test_domains, test_filenames, val_features, val_labels, val_groups, val_domains, val_filenames = load_embedding(cache_file, args.DATA.DATASET)
print('target domain label:',test_domains)
test_set = FEADATASET(
            features=test_features,
            labels=test_labels,
            groups=test_groups,
            domains=test_domains,
            filenames=test_filenames
        )
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.DATA.BATCH_SIZE, shuffle=True)
dataset_domains = args.AUGMENTATION.DOM_LABELS
print("Clients:",dataset_domains,"Client number",len(dataset_domains))

train_set = []
aug_set = []
for i in range(len(dataset_domains)):
    cache_file = f"{args.DATA.SAVE_PATH}/{dataset_domains[i]}/{args.EXP.IMAGE_FEATURES}_{args.EXP.CLIP_PRETRAINED_DATASET}_{args.EXP.CLIP_MODEL.replace('/','_')}.pt"
    train_features, train_labels, train_groups, train_domains, train_filenames, val_features, val_labels, val_groups, val_domains, val_filenames = load_embedding(cache_file, args.DATA.DATASET)
    train_domains = np.full((train_features.shape[0],), i, dtype=int)


    temp_train_set= FEADATASET(
        features=train_features,
        labels=train_labels,
        groups=train_groups,
        domains=train_domains,
        filenames=train_filenames
    )
    temp = torch.utils.data.Subset(temp_train_set, list(range(args.EXP.SHOT)))
    train_set.append(temp)


    print('local domain label:',train_domains)
    augment_features, augment_labels, augment_groups, augment_domains, augment_filenames  = [train_features],[train_labels],[train_groups],[train_domains],[train_filenames]
    for j in range(len(dataset_domains)):
        cache_file = f"{args.DATA.SAVE_PATH}/{dataset_domains[i]}/domain_{dataset_domains[j]}_seed_{args.EXP.SEED}.pt"
        if os.path.exists(cache_file):
            aug_features, aug_labels, aug_groups, aug_domains, aug_filenames = load_aug_embedding(cache_file, args.DATA.DATASET)
            #aug_domains = np.full((aug_features.shape[0],), aug_domains, dtype=int)
            aug_domains = np.full((aug_features.shape[0],), j, dtype=int)
            print('local agu domain label:',aug_domains)    
            num_to_select = min(args.EXP.SHOT, aug_features.shape[0])
            aug_features = aug_features[:num_to_select]
            aug_labels = aug_labels[:num_to_select]
            aug_groups = aug_groups[:num_to_select]
            aug_domains = aug_domains[:num_to_select]
            aug_filenames = aug_filenames[:num_to_select]
            augment_features.append(aug_features)
            augment_labels.append(aug_labels)
            augment_groups.append(aug_groups)
            augment_domains.append(aug_domains)
            augment_filenames.append(aug_filenames)
    final_aug_features = np.concatenate(augment_features, axis=0)
    final_aug_labels = np.concatenate(augment_labels, axis=0)
    final_aug_groups = np.concatenate(augment_groups, axis=0)
    final_aug_domains = np.concatenate(augment_domains, axis=0) 
    final_aug_filenames = np.concatenate(augment_filenames, axis=0)

    aug_set.append(FEADATASET(
        features=final_aug_features,
        labels=final_aug_labels,
        groups=final_aug_groups,
        domains=final_aug_domains,
        filenames=final_aug_filenames
    ))
train_loader = []
aug_loader = []
for i in range(len(dataset_domains)):
    train_loader.append(torch.utils.data.DataLoader(train_set[i], batch_size=args.DATA.BATCH_SIZE, shuffle=True))
    aug_loader.append(torch.utils.data.DataLoader(aug_set[i], batch_size=args.DATA.BATCH_SIZE, shuffle=True))

print('successfully load all datasets')
# 初始化模型
classnames = dh.DATASET_CLASSES[args.DATA.DATASET]
server_model = FaSTPT_Client(classnames=classnames,clip_model=model,
                                      n_ctx_global=args.EXP.N_CTX_GLOBAL, n_ctx_domain=args.EXP.N_CTX_DOMAIN, 
                                    domain_number=len(dataset_domains),client_domain_id=len(dataset_domains),device=device)
client_model = []
for i in range (len(dataset_domains)):
    print("initializing model for client",dataset_domains[i])
    client_model.append(FaSTPT_Client(classnames=classnames,clip_model=model,
                                      n_ctx_global=args.EXP.N_CTX_GLOBAL, n_ctx_domain=args.EXP.N_CTX_DOMAIN, 
                                    domain_number=len(dataset_domains),client_domain_id=i,device=device).to(device))
client_weights = [1/len(dataset_domains) for i in range(len(dataset_domains))]
text_prompts = OmegaConf.to_container(args.EXP.TEXT_PROMPTS, resolve=True)
# client_weights = torch.tensor(client_weights).to(device)
# train (forward type)
best_acc = 0.
loss_fun = nn.CrossEntropyLoss()
for r in range(args.EXP.N_ROUNDS):
    print(f"\n--- Round {r+1}/{args.EXP.N_ROUNDS} ---")
    print(text_prompts)
    for i in range(len(dataset_domains)):

        optimizer = torch.optim.SGD([client_model[i].prompt_learner.global_ctx], lr=args.EXP.LR)
        optimizer2 = torch.optim.SGD(client_model[i].domain_classifier.parameters(), lr=args.EXP.LR2)
        for epoch in range(args.EXP.G_EPOCHS):
            train_loss,train_acc = train_global(client_model[i],train_loader[i],aug_loader[i],optimizer,loss_fun,device)
            print(f'Train round:{r} | {dataset_domains[i]} | Train_G epoches: {epoch} | Train_G Loss: {train_loss:.{4}f} | Train_G Acc: {train_acc:.{4}f}')
            if logout:
                with open(f'{logout}.txt', 'a+') as f_out:
                    f_out.write(f'Train round:{r} | {dataset_domains[i]} | Train_G epoches: {epoch} | Train_G Loss: {train_loss:.{4}f} | Train_G Acc: {train_acc:.{4}f}\n')
            train_loss,train_acc = train_domclassifier(client_model[i],train_loader[i],aug_loader[i],optimizer2,loss_fun,device)
            print(f'Train round:{r} | {dataset_domains[i]} | Train_DC epoches: {epoch} | Train_DC Loss: {train_loss:.{4}f} | Train_DC Acc: {train_acc:.{4}f}')
            if logout:
                with open(f'{logout}.txt', 'a+') as f_out:
                    f_out.write(f'Train round:{r} | {dataset_domains[i]} | Train_DC epoches: {epoch} | Train_DC Loss: {train_loss:.{4}f} | Train_DC Acc: {train_acc:.{4}f}\n')
    Aggregation(server_model,client_model,client_weights=client_weights,device=device)
    for i in range(len(dataset_domains)):     
        optimizer3 = torch.optim.SGD([client_model[i].prompt_learner.domain_ctx], lr=args.EXP.LR)
        for epoch in range(args.EXP.D_EPOCHS):
            train_loss,train_acc = train_domain(client_model[i],train_loader[i],aug_loader[i],optimizer3,loss_fun,device,text_prompts[i][0])
            print(f'Train round:{r} | {dataset_domains[i]} | Train_D epoches: {epoch} | Train_D Loss: {train_loss:.{4}f} | Train_D Acc: {train_acc:.{4}f}')
            if logout:
                with open(f'{logout}.txt', 'a+') as f_out:
                    f_out.write(f'Train round:{r} | {dataset_domains[i]} | Train_D epoches: {epoch} | Train_D Loss: {train_loss:.{4}f} | Train_D Acc: {train_acc:.{4}f}\n')

    P_D_list = torch.stack([client.prompt_learner.domain_ctx.data for client in client_model])

    test_acc = test_infer(server_model,test_loader,device,P_D_list)
    if test_acc > best_acc:
        best_acc = test_acc
        print('savemodel')
        dict = server_model.obtain_prompt(P_D_list=P_D_list)
        print(dict.keys())
        torch.save(dict, os.path.join(output_folder, f"best_model.pt"))

    print(f'Test round:{r} | Test Acc: {test_acc:.{4}f}')
    if logout:
        with open(f'{logout}.txt', 'a+') as f_out:
            f_out.write(f'Test round:{r} | Test Acc: {test_acc:.{4}f}|Best Acc: {best_acc:.{4}f}\n')


