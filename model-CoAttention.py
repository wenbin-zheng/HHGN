import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Transformer import TransformerEncoder
from dgl.nn.pytorch import GATConv
import dgl


class CoAttention(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super(CoAttention, self).__init__()
        self.query = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.key = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.num_heads = num_heads
        self.out_feats = out_feats

    def forward(self, h_audio, h_text):
        Q_audio = self.query(h_audio).view(-1, self.num_heads, self.out_feats)
        K_text = self.key(h_text).view(-1, self.num_heads, self.out_feats)
        attn_weights = torch.einsum('bhd,bhd->bh', Q_audio, K_text)
        attn_weights = F.softmax(attn_weights, dim=-1)
        return attn_weights


class CrossModalGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, K):
        super(CrossModalGAT, self).__init__()
        self.co_attn = CoAttention(in_feats, hidden_feats, num_heads)
        self.gat_audio_to_text = GATConv(in_feats, hidden_feats, num_heads, allow_zero_in_degree=True)
        self.gat_text_to_audio = GATConv(in_feats, hidden_feats, num_heads, allow_zero_in_degree=True)
        self.K = K
        self.fc = nn.Linear(hidden_feats * num_heads * 2, out_feats)

    def forward(self, g, h_audio, h_text):
        attn_weights_audio_to_text = self.co_attn(h_audio, h_text)
        attn_weights_text_to_audio = self.co_attn(h_text, h_audio)

        # Create sparse attention mask
        topk_audio_to_text = torch.topk(attn_weights_audio_to_text, self.K, dim=1)[0]
        topk_text_to_audio = torch.topk(attn_weights_text_to_audio, self.K, dim=1)[0]
        mask_audio_to_text = attn_weights_audio_to_text >= topk_audio_to_text.min(dim=1, keepdim=True)[0]
        mask_text_to_audio = attn_weights_text_to_audio >= topk_text_to_audio.min(dim=1, keepdim=True)[0]

        g.edges['audio-text'].data['attn_mask'] = mask_audio_to_text.float()
        g.edges['text-audio'].data['attn_mask'] = mask_text_to_audio.float()

        h_audio_to_text = self.gat_audio_to_text(g, (h_audio, h_text), edge_weight=mask_audio_to_text.float())
        h_text_to_audio = self.gat_text_to_audio(g, (h_text, h_audio), edge_weight=mask_text_to_audio.float())

        h_audio = F.relu(h_audio_to_text)
        h_text = F.relu(h_text_to_audio)

        h = torch.cat([h_audio, h_text], dim=1)
        return self.fc(h)


class HeGraphModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, K):
        super(HeGraphModel, self).__init__()
        self.cross_modal_gat = CrossModalGAT(in_feats, hidden_feats, out_feats, num_heads, K)

    def forward(self, g, h_audio, h_text):
        out = self.cross_modal_gat(g, h_audio, h_text)
        return out


class CrossmodalNet(nn.Module):
    def __init__(self, inchannels, args) -> None:
        super(CrossmodalNet, self).__init__()

        self.modalities = args.modalities
        n_modals = len(args.modalities)

        layers = nn.ModuleDict()
        for j in self.modalities:
            for k in self.modalities:
                if j == k: continue
                layers_name = j + k
                layers[layers_name] = TransformerEncoder(inchannels, num_heads=args.crossmodal_nheads,
                                                         layers=args.num_crossmodal)
            layers[f'mem_{j}'] = TransformerEncoder(inchannels * (n_modals - 1), num_heads=args.self_att_nheads,
                                                    layers=args.num_self_att)
        self.layers = layers

    # Initialize the CrossModalGAT model
        self.cross_modal_gat = HeGraphModel(inchannels, args.hidden_size, args.hidden_size,
                                                  args.crossmodal_nheads, args.K)

    def forward(self, x_s):

        assert len(x_s) == len(self.modalities), f'{len(x_s)} diff {self.modalities}'

        for j in range(len(x_s)):
            x_s[j] = x_s[j].permute(1, 0, 2)

        out_dict = {}
        for j, x_j in zip(self.modalities, x_s):
            temp = []
            for k, x_k in zip(self.modalities, x_s):
                if j == k: continue
                layer_name = j + k
                out_dict[layer_name] = self.layers[layer_name](x_j, x_k, x_k)
                temp.append(out_dict[layer_name])
            temp = torch.cat(temp, dim=2)
            out_dict[f'mem_{j}'] = self.layers[f'mem_{j}'](temp)
        out = []
        for j in self.modalities:
            out.append(out_dict[f'mem_{j}'])

        out = torch.cat(out, dim=2)

        # Cross-modal GAT forward
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=node_features.size(0)).to(self.device)
        h_audio, h_text = out[:, :self.num_relations // 2], out[:, self.num_relations // 2:]
        out_cross_modal = self.cross_modal_gat(g, h_audio, h_text)

        return out_cross_modal



