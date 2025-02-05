import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse

class HistoricalEdgeModule(nn.Module):
    def __init__(self, node_types, embed_dim, decay_theta=0.1):
        super().__init__()
        self.node_types = node_types 
        self.embed_dim = embed_dim
        self.decay_theta = nn.Parameter(torch.tensor(decay_theta))         
        self.projection = nn.ModuleDict({
            t: nn.Linear(embed_dim, embed_dim, bias=False) 
            for t in node_types
        })
        
    def gamma_decay(self, delta_t):
        return torch.exp(-self.decay_theta * delta_t)  
        
    def compute_base_intensity(self, v_i, v_j, type_i, type_j):

        proj_i = self.projection[type_i](v_i)  # φ(m)
        proj_j = self.projection[type_j](v_j)  # φ(n)
        return -torch.norm(proj_i - proj_j, p=2, dim=-1) 
    
    def forward(self, node_pairs, node_embeds, node_types, event_history):
        """
        imput:
            node_pairs: [(m,n)] 
            node_embeds: {node_id: embedding} 
            node_types: {node_id: type} 
            event_history: { (m,n,r): [t_s1, t_s2...] } 
        output:
            lambda_hist:
        """
        lambda_hist = []
        for m, n in node_pairs:
            type_m = node_types[m]
            type_n = node_types[n]

            gamma_base = self.compute_base_intensity(node_embeds[m], node_embeds[n], type_m, type_n)
            sum_alpha = 0.0
            for (x, y, r), t_list in event_history.items():
                for t_s in t_list:
                    delta_t = self.current_time - t_s 
                    decay = self.gamma_decay(delta_t)
                    sum_alpha += decay  
                    
            lambda_hist.append(gamma_base + sum_alpha)
            
        return torch.stack(lambda_hist)

class TriadicClosureModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def compute_clustering_coeff(self, adj_matrix):
        num_nodes = adj_matrix.size(0)
        tri_counts = torch.diagonal(adj_matrix @ adj_matrix @ adj_matrix) 
        degree = adj_matrix.sum(dim=1) 
        denom = degree * (degree - 1)
        denom[denom == 0] = 1e-6 
        C = 2 * tri_counts / denom
        return C
    
    def forward(self, node_pairs, adj_matrix, node_embeds, event_history):
        C = self.compute_clustering_coeff(adj_matrix)
        lambda_tri = []
        for m, n in node_pairs:
            common_neighbors = torch.where(adj_matrix[m] * adj_matrix[n])[0]
            
            sum_terms = 0.0
            for k in common_neighbors:
                C_k = C[k]
                delta_v = node_embeds[k] - node_embeds[k] 
                g_term = -torch.norm(delta_v, p=2)**2
                sum_terms += C_k * g_term 
                
            lambda_tri.append(sum_terms)
            
        return torch.stack(lambda_tri)

class NeighborInfluenceModule(nn.Module):
    def __init__(self, node_types, embed_dim, num_relations):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_relations = num_relations

        self.W_phi = nn.ModuleDict({ 
            t: nn.Linear(embed_dim, embed_dim, bias=False) 
            for t in node_types
        })
        self.W_zeta = nn.Linear(2*embed_dim, 1, bias=False) 
        
        self.W_beta = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_relations)
        ])
        
    def micro_attention(self, n_embed, s_embed, type_n, type_s, delta_t):

        h_n = self.W_phi[type_n](n_embed)
        h_s = self.W_phi[type_s](s_embed)
        

        combined = torch.cat([h_n, h_s], dim=-1)  
        e_ns = torch.sum(self.W_zeta(combined) * self.gamma_decay(delta_t), dim=-1)
        alpha_ns = F.softmax(e_ns, dim=0)
        return alpha_ns
        
    def meso_aggregation(self, node_embeds, neighbors_dict):
        h_agg = {}
        for r in range(self.num_relations):
            h_r = []
            for n in neighbors_dict[r]:
                h_n = node_embeds[n]
                h_r.append(h_n)
            if len(h_r) > 0:
                h_r = torch.stack(h_r).mean(dim=0) 
                h_agg[r] = self.W_beta[r](h_r)
        return h_agg
    
    def macro_influence(self, h_agg):
        x_tilde = sum(h_agg.values()) / len(h_agg)
        epsilon = torch.sigmoid(x_tilde) 
        return epsilon
    
    def forward(self, node_pairs, node_embeds, node_types, neighbor_data):
        lambda_neighbor = []
        for i, j in node_pairs:
            alpha_i = self.micro_attention(...) 
            alpha_j = self.micro_attention(...)

            h_agg_i = self.meso_aggregation(node_embeds[i], neighbor_data[i])
            h_agg_j = self.meso_aggregation(node_embeds[j], neighbor_data[j])

            epsilon_ij = self.macro_influence(h_agg_i, h_agg_j)
            
            lambda_neighbor.append(epsilon_ij)
            
        return torch.stack(lambda_neighbor)

class MSRL(nn.Module):
    def __init__(self, node_types, num_relations, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

        self.hist_module = HistoricalEdgeModule(node_types, embed_dim)
        self.triadic_module = TriadicClosureModule()
        self.neighbor_module = NeighborInfluenceModule(node_types, embed_dim, num_relations)

        self.node_embeds = nn.ParameterDict({
            nid: nn.Parameter(torch.randn(embed_dim)) 
            for nid in node_types.keys()  
        })
        
        self.q1 = nn.Parameter(torch.zeros(1))
        self.q2 = nn.Parameter(torch.zeros(1))
        
    def forward(self, node_pairs, adj_matrix, event_history, neighbor_data):
        lambda_hist = self.hist_module(node_pairs, self.node_embeds, adj_matrix, event_history)

        lambda_tri = self.triadic_module(node_pairs, adj_matrix, self.node_embeds, event_history)

        lambda_neigh = self.neighbor_module(node_pairs, self.node_embeds, neighbor_data)

        lambda_total = lambda_hist + lambda_tri + lambda_neigh
        lambda_final = torch.exp(lambda_total) 

        Y = self.q1 * lambda_final + self.q2 * ... 
        return torch.sigmoid(Y) 

if __name__ == "__main__":
    node_types = {0: 'author', 1: 'paper'} 
    model = MSRL(node_types, num_relations=4)
    print(model)