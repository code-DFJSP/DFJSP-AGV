import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPsim(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim=128):
        super(MLPsim, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_feats)
        )
    def forward(self, feat, adj=None):
        if feat.size(0) == 0:
            return torch.zeros((1, self.project[-1].out_features), device=feat.device)
        return self.project(feat)

class GATTriple(nn.Module):
    def __init__(self, in_feats_op, in_feats_agv, in_feats_mach, out_feats, num_head=1):
        super(GATTriple, self).__init__()
        self._out = out_feats
        self._heads = num_head
        self.fc_op = nn.Linear(in_feats_op, out_feats * num_head, bias=False)
        self.fc_agv = nn.Linear(in_feats_agv, out_feats * num_head, bias=False)
        self.fc_mach = nn.Linear(in_feats_mach, out_feats * num_head, bias=False)
        self.fc_e1 = nn.Linear(1, out_feats * num_head, bias=False)
        self.fc_e2 = nn.Linear(1, out_feats * num_head, bias=False)
        self.attn_op = nn.Parameter(torch.randn(1, num_head, out_feats))
        self.attn_agv = nn.Parameter(torch.randn(1, num_head, out_feats))
        self.attn_mach = nn.Parameter(torch.randn(1, num_head, out_feats))
        self.attn_e1 = nn.Parameter(torch.randn(1, num_head, out_feats))
        self.attn_e2 = nn.Parameter(torch.randn(1, num_head, out_feats))
        self.leaky = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_op.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_agv.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_mach.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e2.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_op, gain=gain)
        nn.init.xavier_normal_(self.attn_agv, gain=gain)
        nn.init.xavier_normal_(self.attn_mach, gain=gain)
        nn.init.xavier_normal_(self.attn_e1, gain=gain)
        nn.init.xavier_normal_(self.attn_e2, gain=gain)

    def forward(self, op_feat, agv_feat, mach_feat, edge_op_agv, edge_agv_mach):
        if op_feat.size(0) == 0:
            num_agvs = agv_feat.size(0)
            num_machs = mach_feat.size(0)
            device = agv_feat.device
            return torch.zeros((num_agvs, self._out), device=device), torch.zeros((num_machs, self._out), device=device)

        num_ops = op_feat.size(0)
        num_agvs = agv_feat.size(0)
        num_machs = mach_feat.size(0)
        h_op = self.fc_op(op_feat).view(num_ops, self._heads, self._out)
        h_agv = self.fc_agv(agv_feat).view(num_agvs, self._heads, self._out)
        h_mach = self.fc_mach(mach_feat).view(num_machs, self._heads, self._out)
        e1 = edge_op_agv.unsqueeze(-1) if edge_op_agv.dim() == 2 else edge_op_agv
        e2 = edge_agv_mach.unsqueeze(-1) if edge_agv_mach.dim() == 2 else edge_agv_mach
        e1_proj = self.fc_e1(e1).view(num_ops, num_agvs, self._heads, self._out)
        e2_proj = self.fc_e2(e2).view(num_agvs, num_machs, self._heads, self._out)
        el = (h_op * self.attn_op).sum(dim=-1)
        er = (h_agv * self.attn_agv).sum(dim=-1)
        ee = (e1_proj * self.attn_e1).sum(dim=-1)
        score_op_agv = self.leaky(el.unsqueeze(1) + ee + er.unsqueeze(0))
        mask1 = (edge_op_agv != 0)
        score_op_agv_masked = score_op_agv.clone()
        score_op_agv_masked[~mask1.unsqueeze(-1).expand_as(score_op_agv_masked)] = float('-inf')
        alpha_op_to_agv = torch.softmax(score_op_agv_masked.permute(1,0,2), dim=1).permute(1,0,2)
        Wmu = (h_op.unsqueeze(1) + e1_proj) * alpha_op_to_agv.unsqueeze(-1)
        agv_msg = Wmu.sum(dim=0)
        agv_full = torch.sigmoid(agv_msg + h_agv)
        agv_emb = agv_full.mean(dim=1)
        h_agv2 = self.fc_agv(agv_feat).view(num_agvs, self._heads, self._out)
        el2 = (h_agv2 * self.attn_agv).sum(dim=-1)
        er2 = (h_mach * self.attn_mach).sum(dim=-1)
        ee2 = (e2_proj * self.attn_e2).sum(dim=-1)
        score_agv_mach = self.leaky(el2.unsqueeze(1) + ee2 + er2.unsqueeze(0))
        mask2 = (edge_agv_mach != 0)
        score_agv_mach_masked = score_agv_mach.clone()
        score_agv_mach_masked[~mask2.unsqueeze(-1).expand_as(score_agv_mach_masked)] = float('-inf')
        alpha_agv_to_mach = torch.softmax(score_agv_mach_masked.permute(1,0,2), dim=1).permute(1,0,2)
        Wmu2 = (h_agv2.unsqueeze(2) + e2_proj) * alpha_agv_to_mach.unsqueeze(-1)
        mach_msg = Wmu2.sum(dim=0)
        mach_full = torch.sigmoid(mach_msg + h_mach)
        mach_emb = mach_full.mean(dim=1)
        return agv_emb, mach_emb
