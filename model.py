import torch
import torch.nn as nn
import torch.nn.functional as F


class SheafLayer(nn.Module):
    def __init__(self, stalk_dim):
        super().__init__()
        self.stalk_dim = stalk_dim

        self.reflection_net = nn.Sequential(
            nn.Linear(2 * stalk_dim, 64),
            nn.ReLU(),
            nn.Linear(64, stalk_dim)
        )

    def householder_matrix(self, v):
        v_norm = F.normalize(v, dim=1)  # [batch_size, stalk_dim]
        identity = torch.eye(self.stalk_dim, device=v.device).unsqueeze(0).expand(v.size(0), -1, -1)
        outer_product = torch.bmm(v_norm.unsqueeze(2), v_norm.unsqueeze(1))

        return identity - 2 * outer_product

    def compute_degree_matrix_inv_sqrt(self, degrees):
        """For orthogonal maps, D is just degree * identity, so D^(-1/2) is 1/sqrt(degree) * identity"""
        num_nodes, device = degrees.size(0), degrees.device
        D_inv_sqrt = torch.zeros(num_nodes, self.stalk_dim, self.stalk_dim, device=device)
        eye = torch.eye(self.stalk_dim, device=device)

        # For orthogonal F, D_v = deg_v * I, so D_v^(-1/2) = deg_v^(-1/2) * I
        degrees = torch.clamp(degrees, min=1e-5)
        deg_inv_sqrt = 1.0 / torch.sqrt(degrees)

        for v in range(num_nodes):
            D_inv_sqrt[v] = deg_inv_sqrt[v] * eye

        return D_inv_sqrt

    def apply_sheaf_diffusion(self, x, edge_index, D_inv_sqrt, L_sd, batch_size=256):
        src, dst = edge_index
        out = x.clone()
        edge_count = edge_index.size(1)

        for b_start in range(0, edge_count, batch_size):
            b_end = min(b_start + batch_size, edge_count)

            s_batch = src[b_start:b_end]
            d_batch = dst[b_start:b_end]
            L_batch = L_sd[b_start:b_end]

            # Process each edge
            for i in range(b_end - b_start):
                s, d = s_batch[i], d_batch[i]

                # Normalized Laplacian
                L_sd_norm = torch.matmul(torch.matmul(D_inv_sqrt[s], L_batch[i]), D_inv_sqrt[d])
                L_ds_norm = torch.matmul(torch.matmul(D_inv_sqrt[d], L_batch[i].t()), D_inv_sqrt[s])

                out[s] = out[s] - torch.matmul(L_sd_norm, x[d])
                out[d] = out[d] - torch.matmul(L_ds_norm, x[s])

        return out

    def forward(self, x, edge_index):
        num_nodes, device = x.size(0), x.device
        src, dst = edge_index

        v_s = self.reflection_net(torch.cat([x[src], x[dst]], dim=1))
        v_d = self.reflection_net(torch.cat([x[dst], x[src]], dim=1))

        # Orthogonalize matrices
        # [num_edges, stalk_dim, stalk_dim]
        F_sEe = self.householder_matrix(v_s)
        F_dEe = self.householder_matrix(v_d)

        # Degree = count (F^T F = I)
        degrees = torch.zeros(num_nodes, device=device)
        degrees.index_add_(0, src, torch.ones(src.size(0), device=device))
        degrees.index_add_(0, dst, torch.ones(dst.size(0), device=device))

        # Compute D^(-1/2)
        D_inv_sqrt = self.compute_degree_matrix_inv_sqrt(degrees)

        # Compute off-diagonal Laplacian blocks L_sd = -F_sEe^T · F_dEe
        L_sd = -torch.bmm(F_sEe.transpose(1, 2), F_dEe)

        # Apply sheaf diffusion
        out = self.apply_sheaf_diffusion(x, edge_index, D_inv_sqrt, L_sd)

        return F.relu(out)


class SheafNN(nn.Module):
    def __init__(self, in_features, out_features, stalk_dim, num_layers):
        super().__init__()
        # Project to stalk space
        self.input_proj = nn.Linear(in_features, stalk_dim)

        # Sheaf diffusion layers with orthogonal maps
        self.sheaf_layers = nn.ModuleList([SheafLayer(stalk_dim) for _ in range(num_layers)])

        # Project to class space
        self.output_proj = nn.Linear(stalk_dim, out_features)

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))
        for layer in self.sheaf_layers:
            h = layer(h, edge_index)
        return self.output_proj(h)


class GraphCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        src, dst = edge_index
        n = x.size(0)

        # Compute D^{-1/2}
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Message Passing: h_v ← h_v + ∑_{u∈N(v)} D_v^{-1/2} · D_u^{-1/2} · h_u
        out = x.clone()
        for i in range(edge_index.size(1)):
            s, d = src[i], dst[i]
            norm = deg_inv_sqrt[s] * deg_inv_sqrt[d]
            out[d] = out[d] + norm * x[s]
            out[s] = out[s] + norm * x[d]

        return F.relu(self.linear(out))


class GraphCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super().__init__()
        self.input_layer = GraphCNLayer(in_features, hidden_features)
        self.layers = nn.ModuleList([GraphCNLayer(hidden_features, hidden_features) for _ in range(num_layers - 2)])
        self.output_proj = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index):
        h = self.input_layer(x, edge_index)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.output_proj(h)