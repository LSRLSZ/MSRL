import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MSRL
from data_loader import DynamicHeteroDataset
from utils import negative_sampling, compute_metrics, save_checkpoint

# Configuration
parser = argparse.ArgumentParser(description='MSRL Training')
parser.add_argument('--dataset', type=str, required=True, 
                    choices=['imdb', 'aminer', 'yelp', 'dblp'],
                    help='Dataset name')
parser.add_argument('--embed_dim', type=int, default=128,
                    help='Embedding dimension')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate')
parser.add_argument('--negative_ratio', type=int, default=5,
                    help='Negative sampling ratio')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for training')
parser.add_argument('--save_dir', type=str, default='./checkpoints',
                    help='Directory to save checkpoints')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def train():
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = DynamicHeteroDataset(args.dataset)
    node_types = dataset.node_types  # Dictionary of {node_id: type}
    num_relations = dataset.num_relations
    
    # Initialize model
    model = MSRL(
        node_types=node_types,
        num_relations=num_relations,
        embed_dim=args.embed_dim
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    # Create data loader
    train_loader = DataLoader(
        dataset.train_events,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch data
            node_pairs, event_times, relation_types = batch
            node_pairs = node_pairs.to(device)
            event_times = event_times.to(device)
            relation_types = relation_types.to(device)
            
            # Get current graph state
            adj_matrix = dataset.get_adjacency_matrix(event_times)
            neighbor_data = dataset.get_neighbor_data(node_pairs)
            
            # Generate negative samples
            neg_samples = negative_sampling(
                node_pairs=node_pairs,
                node_types=node_types,
                num_negatives=args.negative_ratio,
                device=device
            )
            
            # Forward pass
            optimizer.zero_grad()
            
            # Positive samples prediction
            pos_scores = model(
                node_pairs=node_pairs,
                adj_matrix=adj_matrix,
                event_history=dataset.event_history,
                neighbor_data=neighbor_data
            )
            
            # Negative samples prediction
            neg_scores = model(
                node_pairs=neg_samples,
                adj_matrix=adj_matrix,
                event_history=dataset.event_history,
                neighbor_data=neighbor_data
            )
            
            # Compute loss
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
                
            total_loss = pos_loss + neg_loss + args.l2_lambda * l2_reg
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            # Log batch progress
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1:03d} | Batch {batch_idx:03d} | '
                      f'Loss: {total_loss.item():.4f} | '
                      f'Time: {time.time()-start_time:.2f}s')
        
        # Update learning rate
        scheduler.step(epoch_loss)
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, filename=os.path.join(args.save_dir, f'best_model.pth'))
        
        # Epoch summary
        print(f'Epoch {epoch+1:03d} Summary | '
              f'Avg Loss: {epoch_loss/len(train_loader):.4f} | '
              f'Time: {time.time()-start_time:.2f}s')
    
    print("Training completed!")

if __name__ == '__main__':
    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Start training
    train()