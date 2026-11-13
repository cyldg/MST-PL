import os
import clip
import open_clip
import torch
import numpy as np
import torchvision
import wandb
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
from clip_utils import get_features, evaluate, zeroshot_classifier, get_ensamble_preds, get_pred_overlap, get_nn_metrics, load_embedding
import methods.augmentations

mean=[0.5, 0.5, 0.5]
std = [0.5,0.5,0.5] 
class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(f'{base_path}/{site}_train.pkl', allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(f'{base_path}/{site}_test.pkl', allow_pickle=True)  
        self.domain = site 
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        domain_dict = {'clipart':0, 'infograph':1, 'painting':2, 'quickdraw':3, 'real':4, 'sketch':5}
        self.domain_index = domain_dict[site]
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = os.path.dirname(base_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): 
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)
        domain = self.domain_index
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        # return image, label
        return {
            "image": image,
            "label": label,
            "domain": domain,
            "group": domain, # since we dont have group labels for test set
            "filename": img_path
        }

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
def get_domainnet_dataset(site):
    data_base_path = 'dataset/DomainNet'
    trainset = DomainNetDataset(data_base_path, site, transform=transform_train, train=True)
    valset = DomainNetDataset(data_base_path, site, transform=transform_test, train=False)
    return trainset,valset

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

if args.EXP.WANDB_SILENT:
    os.environ['WANDB_SILENT']="true"

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

if args.EXP.DEBUG:
    # only run for a few epochs
    args.EXP.EPOCHS = 1
    args.AUGMENTATION.EPOCHS = 1

run = wandb.init(project=args.EXP.PROJ, group=args.EXP.ADVICE_METHOD, config=flatten_config(args), allow_val_change=False)
# wandb.save(flags.config)
# wandb.run.log_code(".")

torch.manual_seed(args.EXP.SEED)
np.random.seed(args.EXP.SEED)
random.seed(args.EXP.SEED)

DATASET_NAME = args.DATA.DATASET

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.EXP.IMAGE_FEATURES == 'clip':
    clip_model, preprocess = clip.load(args.EXP.CLIP_MODEL, device)
    model, preprocess = clip.load(args.EXP.CLIP_MODEL, device)
elif args.EXP.IMAGE_FEATURES == 'openclip':
    model, _, preprocess = open_clip.create_model_and_transforms(args.EXP.CLIP_MODEL, pretrained=args.EXP.CLIP_PRETRAINED_DATASET)
    model = model.to(torch.device('cuda'))
    clip_model = model
else:
    model = getattr(torchvision.models, args.EXP.IMAGE_FEATURES)(pretrained=True)
    model = model.to(device)

# # load data
cache_file = f"{args.DATA.SAVE_PATH}/{args.DATA.SOURCE}/{args.EXP.IMAGE_FEATURES}_{args.EXP.CLIP_PRETRAINED_DATASET}_{args.EXP.CLIP_MODEL.replace('/','_')}.pt"
dataset_classes = ['bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra']
dataset_domains = args.AUGMENTATION.DOM_LABELS
if os.path.exists(cache_file):
    print(f"Loading cached embeddings from {cache_file}")
    train_features, train_labels, train_groups, train_domains, train_filenames, val_features, val_labels, val_groups, val_domains, val_filenames = load_embedding(cache_file, args.DATA.DATASET)
    print(train_domains)
else:
    print(f"Computing embeddings and saving to {cache_file}")
    trainset, valset = get_domainnet_dataset(args.DATA.SOURCE)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
    train_features, train_labels, train_groups, train_domains, train_filenames = get_features(train_loader, model, device, model_type=args.EXP.IMAGE_FEATURES)
    val_features, val_labels, val_groups, val_domains, val_filenames = get_features(val_loader, model, device, model_type=args.EXP.IMAGE_FEATURES)
    save_dict = {
        "train_features": train_features, "train_labels": train_labels, "train_groups": train_groups, "train_domains": train_domains, "train_filenames": train_filenames,
        "val_features": val_features, "val_labels": val_labels, "val_groups": val_groups, "val_domains": val_domains, "val_filenames": val_filenames,
        "seed": args.EXP.SEED
    }
    if not os.path.exists(f"{args.DATA.SAVE_PATH}/{args.DATA.SOURCE}"):
        os.makedirs(f"{args.DATA.SAVE_PATH}/{args.DATA.SOURCE}")
    torch.save(save_dict, cache_file)
    print(f"Saved CLIP embeddings to {cache_file}")

if args.METHOD.NORMALIZE:
    train_features /= np.linalg.norm(train_features, axis=-1, keepdims=True)
    val_features /= np.linalg.norm(val_features, axis=-1, keepdims=True)


# Calculate the image features
prompts = list(args.EXP.TEXT_PROMPTS)
if len(prompts) >0 and type(prompts[0]) == omegaconf.listconfig.ListConfig:
    prompts = [list(p) for p in prompts]

neutral_prompts = list(args.EXP.NEUTRAL_TEXT_PROMPTS)
if len(neutral_prompts) >0 and type(neutral_prompts[0]) == omegaconf.listconfig.ListConfig:
    neutral_prompts = [list(p) for p in neutral_prompts]
print("Advice Method", args.EXP.ADVICE_METHOD)
bias_correction = getattr(CLIPTransformations, args.EXP.ADVICE_METHOD)(prompts, clip_model, args, neutral_prompts)


old_val_features, old_val_labels, old_val_groups, old_val_domains, old_val_filenames = val_features, val_labels, val_groups, val_domains, val_filenames

print("old dataset sizes", len(train_features), len(val_features))
num_augmentations = 1
if args.EXP.AUGMENTATION != None and args.EXP.AUGMENTATION != 'None':
    print("Augmenting training set...")
    if "LADS" in args.EXP.AUGMENTATION or 'Directional' in args.EXP.AUGMENTATION:
        augment = getattr(methods.augmentations, args.EXP.AUGMENTATION)(args, train_features, train_labels, train_groups, train_domains, train_filenames, bias_correction.text_embeddings, val_features, val_labels, val_groups, val_domains)
    else:
        augment = getattr(methods.augmentations, args.EXP.AUGMENTATION)(args, train_features, train_labels, train_groups, train_domains, train_filenames, bias_correction.text_embeddings)
    train_features, train_labels, train_domains, train_groups, train_filenames = augment.augment_dataset()
    print(train_domains)
    print("Training set augmented!")
    print("new dataset sizes", len(train_features), len(val_features))


    # 获取所有不同的域
    unique_domains = np.unique(train_domains)

    # 设置保存路径（假设你有适当的配置变量）
    save_path = f"{args.DATA.SAVE_PATH}/{args.DATA.SOURCE}"

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset_domainss = dh.DATASET_DOMAINS[args.DATA.DATASET]
    # 对每个域的数据进行保存
    for domain in unique_domains:
        # 获取当前域的数据
        domain_mask = (train_domains == domain)
        domain_features = train_features[domain_mask]
        domain_labels = train_labels[domain_mask]
        domain_groups = train_groups[domain_mask]
        domain_filenames = train_filenames[domain_mask]
        
        # 创建保存文件的路径
        cache_file = f"{save_path}/domain_{dataset_domainss[int(domain)]}_seed_{args.EXP.SEED}.pt"

        # 创建字典保存当前域的数据
        save_dict = {
            "train_features": domain_features,
            "train_labels": domain_labels,
            "train_groups": domain_groups,
            "train_domains": domain,
            "train_filenames": domain_filenames
        }
        
        # 保存数据到文件
        torch.save(save_dict, cache_file)
        print(f"Saved data for domain {dataset_domainss[int(domain)]} to {cache_file}")

if args.EXP.LOG_NN:
        # features, labels, groups, domains, filenames = np.concatenate([old_val_features, old_test_features]), np.concatenate([old_val_labels, old_test_labels]), np.concatenate([old_val_groups, old_test_groups]), np.concatenate([old_val_domains, old_test_domains]), np.concatenate([old_val_filenames, old_test_filenames])
    cache_file = f"DomainNet_embeddings/{args.AUGMENTATION.DOM_LABELS[0]}/{args.EXP.IMAGE_FEATURES}_{args.EXP.CLIP_PRETRAINED_DATASET}_{args.EXP.CLIP_MODEL.replace('/','_')}.pt"
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        features, labels, groups, domains, filenames, val_features, val_labels, val_groups, val_domains, val_filenames = load_embedding(cache_file, args.DATA.DATASET)   
    print(len(np.unique(train_domains)))
    if len(np.unique(train_domains)) > 1:
        filtered_idxs = np.where(train_domains != train_domains[0])
        sample_features, sample_domains, sample_labels, sample_filenames = train_features[filtered_idxs], train_domains[filtered_idxs], train_labels[filtered_idxs], train_filenames[filtered_idxs]
        sample_idxs = random.sample(list(range(len(sample_filenames))), min((len(train_filenames), 1000)))
        sample_features, sample_domains, sample_labels, sample_filenames = sample_features[sample_idxs], sample_domains[sample_idxs], sample_labels[sample_idxs], sample_filenames[sample_idxs]
    else:
        sample_idxs = random.sample(list(range(len(train_filenames))), min((len(train_filenames), 1000)))
        sample_features, sample_domains, sample_labels, sample_filenames = train_features[sample_idxs], train_domains[sample_idxs], train_labels[sample_idxs], train_filenames[sample_idxs]
    neighbor_domains, neighbor_labels, domain_acc, class_acc, neighbor_samples, prop_unique, mean_cs = get_nn_metrics(sample_features, sample_domains, sample_labels, features, domains, labels)
    plt.rcParams["figure.figsize"] = (20,5)
    f, (axs_orig, axs_new) = plt.subplots(2, 10, sharey=True)
    print(neighbor_samples)
    print(dataset_domainss)
    for i, (original_idx, sample_idx) in enumerate(neighbor_samples):

        print(sample_filenames[original_idx])
        print(filenames[sample_idx])
        axs_orig[i].imshow(Image.open(sample_filenames[original_idx]).resize((224, 224)))
        axs_orig[i].set_title(f"{dataset_domainss[int(sample_domains[int(original_idx)])]} - {sample_labels[int(original_idx)]}")
        axs_orig[i].axis('off')
        axs_new[i].imshow(Image.open(filenames[sample_idx]).resize((224, 224)))
        axs_new[i].set_title(f"{dataset_domainss[int(domains[int(sample_idx)])]} - {labels[int(sample_idx)]}")
        axs_new[i].axis('off')

    wandb.log({"train features NN": wandb.Image(f), "domain consistency acc": domain_acc, "class consistency acc": class_acc, "unique nn": prop_unique})
    # wandb.sklearn.plot_confusion_matrix(sample_domains, neighbor_domains, dataset_domains)
    print("Plotted Nearest Neighbors")

