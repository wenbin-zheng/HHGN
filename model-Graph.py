import torch
import torch.nn as nn
import numpy as np
from .functions import multi_concat, feature_packing
from .model_CoAttention import HeGraphModel


class GraphModel(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, device, args):
        super(GraphModel, self).__init__()

        self.n_modals = len(args.modalities)
        self.wp = args.wp
        self.wf = args.wf
        self.device = device

        print(f"GraphModel --> Edge type: {args.edge_type}")
        print(f"GraphModel --> Window past: {args.wp}")
        print(f"GraphModel --> Window future: {args.wf}")
        edge_temp = "temp" in args.edge_type
        edge_multi = "multi" in args.edge_type

        edge_type_to_idx = {}

        if edge_temp:
            temporal = [-1, 1, 0]
            for j in temporal:
                for k in range(self.n_modals):
                    edge_type_to_idx[str(j) + str(k) + str(k)] = len(edge_type_to_idx)
        else:
            for j in range(self.n_modals):
                edge_type_to_idx['0' + str(j) + str(j)] = len(edge_type_to_idx)

        if edge_multi:
            for j in range(self.n_modals):
                for k in range(self.n_modals):
                    if j != k:
                        edge_type_to_idx['0' + str(j) + str(k)] = len(edge_type_to_idx)

        self.edge_type_to_idx = edge_type_to_idx
        self.num_relations = len(edge_type_to_idx)
        self.edge_multi = edge_multi
        self.edge_temp = edge_temp

        self.gnn = GNN(g_dim, h1_dim, h2_dim, self.num_relations, self.n_modals, args)

       
        self.cross_modal_gat = HeGraphModel(g_dim, h1_dim, h2_dim, args.num_heads, args.K)

    def forward(self, x, lengths):
        node_features = feature_packing(x, lengths)

        node_type, edge_index, edge_type, edge_index_lengths = self.batch_graphify(lengths)

        out_gnn = self.gnn(node_features, node_type, edge_index, edge_type)
        out_gnn = multi_concat(out_gnn, lengths, self.n_modals)

        
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=node_features.size(0)).to(self.device)
        h_audio, h_text = node_features[:, :self.num_relations // 2], node_features[:, self.num_relations // 2:]
        out_cross_modal = self.cross_modal_gat(g, h_audio, h_text)

      
        out = torch.cat((out_gnn, out_cross_modal), dim=-1)

        return out

def batch_graphify(self, lengths):
    node_type, edge_index, edge_type, edge_index_lengths = [], [], [], []
    edge_type_lengths = [0] * len(self.edge_type_to_idx)

    lengths = lengths.tolist()

    sum_length = 0
    total_length = sum(lengths)
    batch_size = len(lengths)

    for k in range(self.n_modals):
        for j in range(batch_size):
            cur_len = lengths[j]
            node_type.extend([k] * cur_len)

    for j in range(batch_size):
        cur_len = lengths[j]

        perms = self.edge_perms(cur_len, total_length)
        edge_index_lengths.append(len(perms))

        for item in perms:
            vertices = item[0]
            neighbor = item[1]
            edge_index.append(torch.tensor([vertices + sum_length, neighbor + sum_length]))

            if vertices % total_length > neighbor % total_length:
                temporal_type = 1
            elif vertices % total_length < neighbor % total_length:
                temporal_type = -1
            else:
                temporal_type = 0
            edge_type.append(self.edge_type_to_idx[str(temporal_type)
                                                   + str(node_type[vertices + sum_length])
                                                   + str(node_type[neighbor + sum_length])])

        sum_length += cur_len

    node_type = torch.tensor(node_type).long().to(self.device)
    edge_index = torch.stack(edge_index).t().contiguous().to(self.device)  # [2, E]
    edge_type = torch.tensor(edge_type).long().to(self.device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(self.device)  # [B]

    return node_type, edge_index, edge_type, edge_index_lengths


