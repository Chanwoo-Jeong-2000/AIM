import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

def data(dataset_name):
    dataset = dataset_name

    data = HeteroData()
    node_types = ['user', 'item']
    attr_names = ['edge_index', 'edge_label_index']

    df_train = pd.read_csv('dataset/'+dataset+'/train.csv')
    df_test = pd.read_csv('dataset/'+dataset+'/test.csv')

    data[node_types[0]].num_nodes = len(np.unique(df_train.userId.values))
    data[node_types[1]].num_nodes = len(np.unique(df_train.itemId.values))

    edge_index = torch.tensor(np.stack([df_train['userId'].values, df_train['itemId'].values]))
    data['user', 'rates', 'item'][attr_names[0]] = edge_index
    data['item', 'rated_by', 'user'][attr_names[0]] = edge_index.flip([0])

    edge_label_index = torch.tensor(np.stack([df_test['userId'].values, df_test['itemId'].values]))
    data['user', 'rates', 'item'][attr_names[1]] = edge_label_index

    return data