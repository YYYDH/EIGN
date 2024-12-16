import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
from itertools import repeat
import networkx as nx
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Batch, Data
from tqdm import tqdm
import warnings

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')


def _cos_sim(x, y):
    cos_sim = F.cosine_similarity(x, y, dim=1).view(-1, 1)
    return cos_sim


def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                  one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))


def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        graph.add_edge(i, j)


def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index


def inter_graph(ligand, pocket, dis_threshold=5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j + atom_num_l)

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter


def get_x_and_edge_index(ligand, pocket, x_l, x_p, edge_index_l, edge_index_p, dis_threshold=5.):
    lenth_l = x_l.shape[0]
    lenth_p = x_p.shape[0]
    num_chunks = lenth_l  # 希望分割的块数
    chunk_size = lenth_p // num_chunks  # 计算每个块的理想大小

    chunks = []
    start = 0
    for i in range(num_chunks - 1):
        end = start + chunk_size
        chunks.append(x_p[start:end])
        start = end

    # 最后一个块可能大小不一样
    chunks.append(x_p[start:])

    edge_index_l_copy = edge_index_l.clone()
    edge_index_p_copy = edge_index_p.clone()
    alternating = []

    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    node_idx_copy = np.array(node_idx)
    node_idx_copy = torch.tensor(node_idx_copy)

    for i in range(lenth_l):
        alternating.append(x_l[i].unsqueeze(0))
        alternating.append(chunks[i])
        # 更新对应的edge_index
        len_chunks = len(chunks[i])
        last_len_chunks = len(chunks[i - 1])
        edge_index_l_copy[edge_index_l == i] = i + i * last_len_chunks
        node_idx_copy[0][node_idx[0] == i] = i + i * last_len_chunks
        for j in range(len_chunks):
            edge_index_p_copy[edge_index_p == (i * last_len_chunks + j)] = i + i * last_len_chunks + j + 1
            node_idx_copy[1][node_idx[1] == (i * last_len_chunks + j)] = i + i * last_len_chunks + j + 1

    x = torch.cat(alternating, dim=0)
    edge_index = torch.cat([edge_index_l_copy, edge_index_p_copy], dim=1)
    edge_index = edge_index[:, edge_index[0].argsort()]

    if edge_index.max() > lenth_l + lenth_p:
        print(edge_index.max())
    if node_idx_copy.max() > lenth_l + lenth_p:
        print(node_idx_copy.max())
    if x.shape[0] != lenth_l + lenth_p:
        print('error', x.shape[0], lenth_l + lenth_p, len(chunks), lenth_l)
    return x, edge_index, node_idx_copy


def mols2graphs(complex_path, label, save_path, dis_threshold=5.):
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p = mol2graph(pocket)
    # x, edge_index_intra, edge_index_inter = get_x_and_edge_index(ligand, pocket, x_l, x_p, edge_index_l, edge_index_p, dis_threshold=dis_threshold)
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
    edge_index = torch.cat([edge_index_intra, edge_index_inter], dim=-1)
    edge_index = edge_index[:, edge_index[0].argsort()]
    y = torch.FloatTensor([label])
    pos = torch.concat([pos_l, pos_p], dim=0)
    split = torch.cat([torch.zeros((atom_num_l,)), torch.ones((atom_num_p,))], dim=0)
    edge_index_p += atom_num_l

    # data = Data(x=x, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)
    data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)
    # data = Data(x=x, edge_index=edge_index, edge_index_inter=edge_index_inter,
    #             edge_index_intra=edge_index_intra, y=y, pos=pos, split=split)

    # torch.save(data, save_path)
    return data


class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


class ProcessedData:
    def __init__(self, data, slices):
        self.data = data
        self.slices = slices


class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, data_dir, data_df, data_root, dataset, dis_threshold=5, num_process=4, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.data_root = data_root
        self.dataset = dataset
        self.dis_threshold = dis_threshold
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self.slices = None
        self.data_list = []
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        dis_thresholds = repeat(self.dis_threshold, len(data_df))
        dist_thresholds_list = []

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        if self._has_cache():
            self._load()
            return
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)
            dist_thresholds_list.append(self.dis_threshold)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            # pool = multiprocessing.Pool(self.num_process)
            # pool.starmap(mols2graphs,
            #              zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
            # pool.close()
            # pool.join()
            for i in tqdm(range(len(complex_path_list))):
                graph = mols2graphs(complex_path_list[i], pKa_list[i], graph_path_list[i], dist_thresholds_list[i])
                self.data_list.append(graph)
            # self.slices = self.get_slices()
            self._save()

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        return self.data_list[idx]

    def _has_cache(self):
        return os.path.exists(os.path.join(self.data_root, f"{self.dataset}.pkl"))

    def _save(self):
        print('Saving...')
        with open(os.path.join(self.data_root, f"{self.dataset}.pkl"), 'wb') as f:
            pickle.dump(self.data_list, f)

    def _load(self):
        print('Loading...')
        with open(os.path.join(self.data_root, f"{self.dataset}.pkl"), 'rb') as f:
            self.data_list = pickle.load(f)

    # def get_slices(self):
    #     if self.slices is None:
    #         cumsum = [0] + [data.num_nodes for data in self.data_list]
    #         cumsum_edges = [0] + [data.edge_index.shape[1] for data in self.data_list]
    #         self.slices = {
    #             'x': cumsum[:-1],
    #             'edge_index': cumsum_edges[:-1],
    #             'edge_attr': cumsum_edges[:-1],
    #             # Add more slices if you have additional attributes
    #         }
    #     return self.slices

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.data_df)


def create_compare_graph(data_dir, save_dir, dataset):
    with open(data_dir, 'rb') as f:
        data_list, slices = pickle.load(f)

    data_B = Data(x=torch.cat([data.x for data in data_list], dim=0),
                  edge_index=torch.cat([data.edge_index for data in data_list], dim=1),
                  edge_index_intra=torch.cat([data.edge_index_intra for data in data_list], dim=1),
                  edge_index_inter=torch.cat([data.edge_index_inter for data in data_list], dim=1),
                  y=torch.cat([data.y for data in data_list], dim=0),
                  pos=torch.cat([data.pos for data in data_list], dim=0),
                  split=torch.cat([data.split for data in data_list], dim=0)
                  )
    data_to_save = [data_B, slices]
    torch.save(data_to_save, save_dir + dataset + '.pt')


if __name__ == '__main__':
    data_root = './dataset'
    toy_dir = os.path.join('./dataset/train')
    toy_df = pd.read_csv(os.path.join(data_root, "train.csv"))
    dataset = 'train'
    toy_set = GraphDataset(toy_dir, toy_df, data_root, dataset, dis_threshold=5, create=True)
    train_loader = PLIDataLoader(toy_set, batch_size=128, shuffle=True, num_workers=4)
    # create_compare_graph(os.path.join(data_root, f"{dataset}.pkl"), data_root, dataset)
    for data in train_loader:
        print(data.edge_index_inter)
        raise False

# %%
