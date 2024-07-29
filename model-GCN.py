import torch
# import torch.nn as nn
# from torch_geometric.nn import RGCNConv, TransformerConv
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv, SAGEConv
class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, num_modals, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_modals = num_modals

        if args.gcn_conv == "rgcn":
            print("GNN --> Use RGCN")
            self.conv1 = RGCNConv(g_dim, h1_dim, num_relations)
            self.bn1 = nn.BatchNorm1d(h1_dim)

        if args.use_graph_transformer:
            print("GNN --> Use Graph Transformer")
            self.conv2 = TransformerConv(h1_dim, h2_dim, heads=args.graph_transformer_nheads, concat=True)
            self.bn2 = nn.BatchNorm1d(h2_dim * args.graph_transformer_nheads)
            self.h2_dim_transformer = h2_dim * args.graph_transformer_nheads  # 记录 transformer 的输出维度

        if args.use_sage_conv:
            print("GNN --> Use GraphSAGE")
            input_dim = h1_dim if not args.use_graph_transformer else self.h2_dim_transformer
            self.conv3 = SAGEConv(input_dim, h2_dim)
            self.bn3 = nn.BatchNorm1d(h2_dim)

    def forward(self, node_features, node_type, edge_index, edge_type):
        residual = node_features  # Initial residual for the first layer

        if self.args.gcn_conv == "rgcn":
            x = self.conv1(node_features, edge_index, edge_type)
            x = nn.functional.leaky_relu(self.bn1(x))
            if x.size(1) != residual.size(1):
                residual = nn.functional.pad(residual, (0, x.size(1) - residual.size(1)))
            x = x + residual  # Residual connection
            residual = x  # Update residual

        if self.args.use_graph_transformer:
            x = self.conv2(x, edge_index)
            x = nn.functional.leaky_relu(self.bn2(x))
            if x.size(1) != residual.size(1):
                residual = nn.functional.pad(residual, (0, x.size(1) - residual.size(1)))
            x = x + residual  # Residual connection
            residual = x  # Update residual

        if self.args.use_sage_conv:
            x = self.conv3(x, edge_index)
            x = nn.functional.leaky_relu(self.bn3(x))
            if x.size(1) != residual.size(1):
                residual = nn.functional.pad(residual, (0, x.size(1) - residual.size(1)))
            x = x + residual  # Residual connection

        return x






        
