import argparse
import json
import os
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from model import MSRL
from data_loader import DynamicHeteroDataset
from utils import load_checkpoint, negative_sampling

# Configuration
parser = argparse.ArgumentParser(description='MSRL Evaluation')
parser.add_argument('--dataset', type=str, required=True,
                    choices=['imdb', 'aminer', 'yelp', 'dblp'],
                    help='Dataset to evaluate')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to trained model checkpoint')
parser.add_argument('--task', type=str, required=True,
                    choices=['link_prediction', 'node_classification'],
                    help='Evaluation task type')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Evaluation batch size')
parser.add_argument('--k_values', type=int, nargs='+', default=[1, 2],
                    help='Top-k values for evaluation')
parser.add_argument('--output_dir', type=str, default='./results',
                    help='Directory to save evaluation results')
args = parser.parse_args()

def evaluate_link_prediction(model, dataset, device):
    """Evaluate link prediction task (Section 4.2)"""
    model.eval()
    test_events = dataset.test_events
    all_scores = []
    all_labels = []
    
    # Get final graph state
    final_time = dataset.max_timestamp
    adj_matrix = dataset.get_adjacency_matrix(final_time)
    
    # Generate negative samples
    neg_samples = negative_sampling(
        node_pairs=test_events[:, :2],
        node_types=dataset.node_types,
        num_negatives=args.negative_ratio,
        device=device
    )
    
    # Combine positive and negative samples
    all_pairs = torch.cat([test_events[:, :2], neg_samples])
    all_labels = torch.cat([
        torch.ones(len(test_events)),
        torch.zeros(len(neg_samples))
    ]).to(device)
    
    # Batch processing
    with torch.no_grad():
        for i in range(0, len(all_pairs), args.batch_size):
            batch_pairs = all_pairs[i:i+args.batch_size]
            neighbor_data = dataset.get_neighbor_data(batch_pairs)
            
            scores = model(
                node_pairs=batch_pairs,
                adj_matrix=adj_matrix,
                event_history=dataset.event_history,
                neighbor_data=neighbor_data
            )
            all_scores.append(scores.cpu())
    
    all_scores = torch.cat(all_scores)
    results = {}
    
    # Calculate metrics for different k values
    for k in args.k_values:
        # Get top-k predictions
        topk_values, topk_indices = torch.topk(all_scores, k=k, dim=0)
        
        # Convert to binary predictions
        preds = torch.zeros_like(all_scores)
        preds[topk_indices] = 1
        
        # Calculate metrics
        results[f'Top-{k}'] = {
            'Micro-F1': f1_score(all_labels.cpu(), preds, average='micro'),
            'Recall': recall_score(all_labels.cpu(), preds),
            'Precision': precision_score(all_labels.cpu(), preds)
        }
    
    return results

def evaluate_node_classification(model, dataset, device):
    """Evaluate node classification task (Section 4.3)"""
    model.eval()
    test_nodes = dataset.test_nodes
    labels = dataset.node_labels
    embeddings = model.node_embeds
    
    # Get class distributions
    unique_classes = torch.unique(labels)
    results = {}
    
    # Create feature matrix and label vector
    X = torch.stack([embeddings[n] for n in test_nodes])
    y = torch.stack([labels[n] for n in test_nodes])
    
    # Split into train/test (80-20) for evaluation
    indices = torch.randperm(len(test_nodes))
    split_idx = int(0.8 * len(indices))
    
    # Use simple logistic regression classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[indices[:split_idx]].cpu().numpy(), 
            y[indices[:split_idx]].cpu().numpy())
    
    # Predict on test split
    preds = clf.predict(X[indices[split_idx:]].cpu().numpy())
    probs = clf.predict_proba(X[indices[split_idx:]].cpu().numpy())
    y_test = y[indices[split_idx:]].cpu().numpy()
    
    # Calculate metrics
    results['Classification'] = {
        'AUC-ROC': roc_auc_score(y_test, probs, multi_class='ovo'),
        'Macro-F1': f1_score(y_test, preds, average='macro'),
        'Micro-F1': f1_score(y_test, preds, average='micro')
    }
    
    return results

def main():
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = DynamicHeteroDataset(args.dataset)
    
    # Initialize model
    model = MSRL(
        node_types=dataset.node_types,
        num_relations=dataset.num_relations,
        embed_dim=128  # Should match training dimension
    ).to(device)
    
    # Load trained weights
    load_checkpoint(model, args.checkpoint, device)
    
    # Run evaluation
    if args.task == 'link_prediction':
        results = evaluate_link_prediction(model, dataset, device)
    elif args.task == 'node_classification':
        results = evaluate_node_classification(model, dataset, device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{args.dataset}_{args.task}_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f'\nEvaluation Results for {args.dataset} ({args.task}):')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()