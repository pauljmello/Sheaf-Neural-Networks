<div align="center">
  <img src="images/sheafs.png" alt="Colorful Sheaf Illustration">
</div>

# Sheaf Neural Networks (SNNs)

## Sheafs

A cellular sheaf on a graph G = (V, E) assigns vector spaces F(v), F(e) to vertices and edges with linear restriction maps F(v→e): F(v) → F(e) that define how information flows. 
Think of sheaves as adding local vector spaces, the stalks, to graphs and projections between those graph points, allowing data to flow like wind across the field of sheaves while maintaining local independence.
For more information on the structure and my implementation of Sheaf Neural Networks, please read the attached Sheaf Research paper.

## Core Concepts

### Stalks
Vector spaces attached to nodes and edges, providing local feature spaces for independent representations:
```python
# Project input features to stalk space
h = F.relu(self.input_proj(x))  # [N, stalk_dim]
```

### Restriction Maps
Linear transformations between stalks govern how information passes to nodes:
```python
# Generate orthogonal maps using Householder transformations
F_sEe = self.householder_matrix(v_s)  # [num_edges, stalk_dim, stalk_dim]
```

### Sheaf Laplacian (L_F)
Generalizes the graph Laplacian through restriction mappings:
```
L_F = δᵀδ, where δ is the coboundary map
L_F[v,u] = -F(v→e)ᵀ F(u→e) for edge e connecting v and u
```

### Sheaf Diffusion
Information flow guided by the sheaf Laplacian:
```
Ẋ(t) = -L_F X(t)
```

## Implementation

Orthogonal restriction maps for better generalization:
```python
# Compute off-diagonal Laplacian blocks
L_sd = -torch.bmm(F_sEe.transpose(1, 2), F_dEe)

# Apply normalized sheaf diffusion
L_sd_norm = torch.matmul(torch.matmul(D_inv_sqrt[s], L_batch[i]), D_inv_sqrt[d])
out[s] = out[s] - torch.matmul(L_sd_norm, x[d])
```

## Results

![Sheaf vs Graph Network Comparison](images/model_comparison.png "SheafNN vs GraphCN Networks Comparison")

In our experiments, we average over three runs, and set the following hyperparameters: layer count = 2, optimizer learning rate and weight decay = 0.01, and set the graph convolution hidden features to 64.
Our experiments demonstrate that SheafNN can offer better training and test accuracies, while also providing better losses over the course of training.

## Conclusion

In this work, SNNs were expanded to the multi-layer setting with the applied Sheaf Diffusion strategy across the cellular sheaves and demonstrated that SNNs outperform GraphCN in a variety of ways. These strategies were utilized, but were developed in prior works. 
These prior works include the Sheaf Laplacian, which generalizes the graph Laplacian through restriction mappings, and the Sheaf Diffusion, which guides information flow through the Sheaf Laplacian. These methods help to significantly boost the generalization capabilities through abstractions.

SNNs provide a very unique approach to modeling complex data. By extending and improving these approaches SNNs will excel in complex domains like drug discovery, materials science, social network analysis, financial systems, and knowledge graphs where heterophily and expressivity are key.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pjmSNN2025,
  author = {Paul J Mello},
  title = {Sheaf Neural Networks},
  url = {https://github.com/pauljmello/Sheaf-Neural-Networks},
  year = {2025},
}
```

## References

[1] Hansen, J., & Gebhart, T. (2020). Sheaf Neural Networks. NeurIPS arXiv:2012.06333.

[2] Bodnar, C., et al. (2022). Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs. NeurIPS 2022. arXiv:2202.04579.

[3] Barbero, F., et al. (2022). Sheaf Neural Networks with Connection Laplacians. ICML. arXiv:2206.08702.
