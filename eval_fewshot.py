import torch
from torch import nn
import random
import numpy as np
from datasets.custom_folder import ScanobjNNcolor, info_collate_fn
from models.MAE3Dsparse_finetune import SWITransformerBase
import argparse
import sklearn
from sklearn.svm import SVC
import glob
import torch.nn.functional as F
from tqdm import tqdm 

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=5, metavar='N',
                        help='Num of nearest neighbors to use')
parser.add_argument('--dataset', type=str, default='scanobjectnn', metavar='N',
                        choices=['scanobjectnn'],
                        help='Dataset to evaluate')
parser.add_argument('--n_runs', type=int, default=10,
                        help='Num of few-shot runs')
parser.add_argument('--k_way', type=int, default=5,
                        help='Num of classes in few-shot')
parser.add_argument('--m_shot', type=int, default=10,
                        help='Num of samples in one class')
parser.add_argument('--n_query', type=int, default=20,
                        help='Num of query samples in one class')
parser.add_argument('--model', type=str, default='', metavar='N',
                        help='Pretrained model')
args = parser.parse_args()

device = torch.device("cuda")

#Try to load models
if args.model == 'SWITransformerBase' and args.dataset == 'scanobjectnn':
    model_path = '/xxxxx/xxxxx.pth'
    class model_config:
        in_chans=12
        patch_num=512
        num_classes=15
        drop_path_rate=0.0
        global_pool=True
        linear_probe=False 
        smoothing=False
        
    model = SWITransformerBase(model_config)
    model.load_from_ckpts(model_path)
    model.head = nn.Identity()
    model = model.to(device)

if args.dataset == 'scanobjectnn':
    # ScanObjectNN - Few Shot Learning
    class Config:
        def __init__(self, subset):
            self.ROOT = '/xxxxx/ScanObjectNN/object_dataset'
            self.subset = subset
            self.with_bg = True
            self.npoints = 100000
            self.patch_size = 16
            self.patch_num = 512
            self.space_type = 'normal'
            self.no_aug = True
    train_config = Config('train')
    test_config = Config('test')

    train_set = ScanobjNNcolor(train_config)
    test_set = ScanobjNNcolor(test_config)
    data_train_indx = np.arange(0, len(train_set))
    
    data_train = []
    label_train = []
    for i, info in enumerate(train_set):
        data_train.append(info)
        label_train.append(info['target'])

    n_cls = 15
else:
    raise NotImplementedError("Unknown dataset!")

label_idx = {}
for key in range(n_cls):
    label_idx[key] = []
    for i, label in enumerate(label_train):
        # if label[0] == key:
        if label == key:
            label_idx[key].append(i)

acc = []
for run in tqdm(range(args.n_runs)):
    k = args.k_way ; m = args.m_shot ; n_q = args.n_query

    k_way = random.sample(range(n_cls), k)

    data_support = [] ; label_support = [] ; data_query = [] ; label_query = []
    for i, class_id in enumerate(k_way):
        support_id = random.sample(label_idx[class_id], m)
        query_id = random.sample(list(set(label_idx[class_id]) - set(support_id)), n_q)

        pc_support_id = data_train_indx[support_id]
        pc_query_id = data_train_indx[query_id]
        data_support.append(pc_support_id)
        label_support.append(i * np.ones(m))
        data_query.append(pc_query_id)
        label_query.append(i * np.ones(n_q))

    data_support = np.concatenate(data_support)
    label_support = np.concatenate(label_support)
    data_query = np.concatenate(data_query)
    label_query = np.concatenate(label_query)

    feats_train = []
    labels_train = []
    model = model.eval()

    for i in range(k * m):
        data = data_train[data_support[i]]
        data = info_collate_fn([data])
        label = int(label_support[i])
        for key, value in data[1].items():
            data[1][key] = value.cuda()
        with torch.no_grad():
            feat = model(data)[0, :]
        feat = feat.detach().cpu().numpy().tolist()
        feats_train.append(feat)
        labels_train.append(label)

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)

    feats_test = []
    labels_test = []

    for i in range(k * n_q):
        data = data_train[data_query[i]]
        data = info_collate_fn([data])
        label = int(label_query[i])
        for key, value in data[1].items():
            data[1][key] = value.cuda()
        with torch.no_grad():
            feat = model(data)[0, :]
        feat = feat.detach().cpu().numpy().tolist()
        feats_test.append(feat)
        labels_test.append(label)

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats_train)
    model_tl = SVC(kernel ='linear')
    model_tl.fit(scaled, labels_train)
    # model_tl.fit(feats_train, labels_train)
    
    test_scaled = scaler.transform(feats_test)
    
    # accuracy = model_tl.score(feats_test, labels_test) * 100
    accuracy = model_tl.score(test_scaled, labels_test) * 100
    acc.append(accuracy)

    # print(f"C = {c} : {model_tl.score(test_scaled, labels_test)}")
    # print(f"Run - {run + 1} : {accuracy}")
    
print(f'{np.mean(acc)} +/- {np.std(acc)}')