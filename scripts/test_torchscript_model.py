import torch

def test_torchscript_model(model_path):
    # Load the TorchScript model
    model = torch.jit.load(model_path)
    
    # Example dummy input based on inferred input structure
    # Assumed feature dimensions (adjusted)
    num_nodes = 100  # Number of nodes
    num_edges = 300  # Number of edges
    in_feats = 8  # Adjusted number of input node features
    
    # Creating dummy input with necessary keys
    dummy_input = {
        'x': {
            'u': torch.randn(num_nodes, in_feats),  # Example: 100 nodes with 8 features each
            'v': torch.randn(num_nodes, in_feats),  # Example: 100 nodes with 8 features each
            'y': torch.randn(num_nodes, in_feats)   # Example: 100 nodes with 8 features each
        },
        'edge_index_plane': {
            'u': torch.randint(0, num_nodes, (2, num_edges)),  # Example: 300 edges
            'v': torch.randint(0, num_nodes, (2, num_edges)),  # Example: 300 edges
            'y': torch.randint(0, num_nodes, (2, num_edges))   # Example: 300 edges
        },
        'edge_index_nexus': {
            'u': torch.randint(0, num_nodes, (2, num_edges)),  # Example: 300 edges
            'v': torch.randint(0, num_nodes, (2, num_edges)),  # Example: 300 edges
            'y': torch.randint(0, num_nodes, (2, num_edges))   # Example: 300 edges
        },
        'nexus': torch.randn(num_nodes, in_feats),  # Example: 100 nodes with 8 features each
        'batch': {
            'u': torch.randint(0, 10, (num_nodes,)),  # Example: batch info for 100 nodes
            'v': torch.randint(0, 10, (num_nodes,)),  # Example: batch info for 100 nodes
            'y': torch.randint(0, 10, (num_nodes,))   # Example: batch info for 100 nodes
        }
    }
    
    # Extracting the individual arguments
    x = dummy_input['x']
    edge_index_plane = dummy_input['edge_index_plane']
    edge_index_nexus = dummy_input['edge_index_nexus']
    nexus = dummy_input['nexus']
    batch = dummy_input['batch']
    
    # Run a forward pass to check if the model works
    with torch.no_grad():
        output = model(x, edge_index_plane, edge_index_nexus, nexus, batch)
    
    print("Model output:", output)

if __name__ == '__main__':
    model_path = '/home/stokes/NuGraph_v24_4/scripts/model.pt'  # Change this to the actual path of your model.pt
    test_torchscript_model(model_path)

