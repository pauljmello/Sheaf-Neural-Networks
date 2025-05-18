import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from model import SheafNN, GraphCN


def create_dataset(num_nodes, num_classes, device='cpu'):
    y = torch.randint(0, num_classes, (num_nodes,)).to(device)

    # Generate features with classes
    x = torch.randn(num_nodes, 4, device=device)
    for i in range(num_nodes):
        x[i] += y[i] * torch.ones(4, device=device)

    # Generate edges with mixed homophily
    edge_list = []
    for i in range(num_nodes):
        connections = np.random.choice([j for j in range(num_nodes) if j != i], size=np.random.randint(2, 4), replace=False)
        for j in connections:
            edge_list.append([i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()

    # Create  masks
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[indices[:int(0.8 * num_nodes)]] = True
    test_mask = ~train_mask

    return x, edge_index, y, train_mask, test_mask


def train(model, x, edge_index, y, train_mask, test_mask, epochs, lr, plot_every):
    train_accs, test_accs, losses = [], [], []

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            losses.append(loss.item())

        if epoch % plot_every == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Train={train_acc:.4f}, Test={test_acc:.4f}")

    return {'train_accs': train_accs, 'test_accs': test_accs, 'losses': losses, 'train_acc': train_accs[-1], 'test_acc': test_accs[-1]}


def plot_results(sheaf_results, gcn_results):
    os.makedirs('results', exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    models = ['SheafNN', 'GraphCN']
    train_accs = [sheaf_results['train_acc'], gcn_results['train_acc']]
    test_accs = [sheaf_results['test_acc'], gcn_results['test_acc']]
    x_pos = range(len(models))
    width = 0.35

    ax[0].bar([p - width / 2 for p in x_pos], train_accs, width, label='Train')
    ax[0].bar([p + width / 2 for p in x_pos], test_accs, width, label='Test')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(models)
    ax[0].legend()
    ax[0].set_ylim(0, 1.0)

    # Loss
    ax[1].plot(sheaf_results['losses'], label='SheafNN')
    ax[1].plot(gcn_results['losses'], label='GraphCN')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Training Loss')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Synthetic Dataset Hyperparameters
    num_nodes = 200
    num_classes = 5

    # Create dataset
    print("Creating dataset...")
    x, edge_index, y, train_mask, test_mask = create_dataset(num_nodes=num_nodes, num_classes=num_classes, device=str(device))
    print(f"Dataset: {x.size(0)} nodes, {edge_index.size(1)} edges, {int(y.max()) + 1} classes")

    # SheafNN Hyperparameters
    # Care Overfitting: Num Layers > Stalk Dims
    num_layers = 2
    stalk_dim = max(2, int(num_classes / 3))

    # GraphCN Hyperparameters
    graph_hidden_features = 64

    # General Hyperparameters
    in_features = x.size(1)
    out_features = int(y.max() + 1)
    epochs = 250
    lr = 0.01
    plot_every = 10

    # Train SheafNN
    print("\nTraining Sheaf Neural Network...")
    sheaf_model = SheafNN(in_features, out_features, stalk_dim, num_layers).to(device)
    sheaf_results = train(sheaf_model, x, edge_index, y, train_mask, test_mask, epochs, lr, plot_every)

    # Train GraphCN
    print("\nTraining Graph Convolutional Network...")
    gcn_model = GraphCN(in_features, graph_hidden_features, out_features, num_layers).to(device)
    gcn_results = train(gcn_model, x, edge_index, y, train_mask, test_mask, epochs, lr, plot_every)

    # Plot comparisons
    plot_results(sheaf_results, gcn_results)

    # Print final results
    print("\nFinal results:")
    print(f"- SheafNN: Train={sheaf_results['train_acc']:.4f}, Test={sheaf_results['test_acc']:.4f}")
    print(f"- GraphCN: Train={gcn_results['train_acc']:.4f}, Test={gcn_results['test_acc']:.4f}")


if __name__ == "__main__":
    main()