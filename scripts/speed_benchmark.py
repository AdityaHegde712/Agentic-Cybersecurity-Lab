import time
import torch
import torch.nn as nn
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Define LSTM-AE Model
class LSTMAE(nn.Module):
    def __init__(self, input_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(64, 32, batch_first=True)
        
        self.decoder_lstm1 = nn.LSTM(32, 64, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(64, input_dim, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, (h, c) = self.encoder_lstm1(x)
        out, (h, c) = self.encoder_lstm2(out)
        
        # We take the final hidden state or repeat the representation
        # For sequence-to-sequence reconstruction, we repeat the latent representation
        latent = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        
        out, (h, c) = self.decoder_lstm1(latent)
        reconstruction, _ = self.decoder_lstm2(out)
        return reconstruction

# Define Anomaly Transformer Proxy (Transformer Encoder)
class AnomalyTransformerProxy(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x_emb = self.embedding(x)
        out = self.transformer(x_emb)
        reconstruction = self.output_proj(out)
        return reconstruction

# Define Graph Neural Network (GNN) Proxy for sensor dependencies
# Models spatial correlations across D sensors, embedding temporal windows of length W
class GNNProxy(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim # Number of sensors D
        self.seq_len = seq_len     # Time window W
        
        # Temporal embedding per sensor: embeds W features to hidden_dim
        self.temp_embedding = nn.Linear(seq_len, hidden_dim)
        
        # Fully connected Graph Attention Layers (modeled as Self-Attention across nodes/sensors)
        self.node_attn1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.node_attn2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Output project per sensor back to 1 value (reconstruction of the latest timestamp)
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, W, D) -> transpose to (batch, D, W) to treat each sensor as a node
        x = x.transpose(1, 2)
        
        # Embed temporal window
        node_features = self.temp_embedding(x) # (batch, D, hidden_dim)
        
        # Graph Attention Layer 1
        out, _ = self.node_attn1(node_features, node_features, node_features)
        # Graph Attention Layer 2
        out, _ = self.node_attn2(out, out, out)
        
        # Predict/Reconstruct
        pred = self.output_proj(out) # (batch, D, 1)
        return pred.squeeze(-1) # (batch, D)

# Run benchmark function
def run_benchmark(device, model_class, input_dim, seq_len, batch_size=256, warmup_epochs=10, run_epochs=50):
    if model_class == GNNProxy:
        # GNN takes shape (batch, seq_len, input_dim) in our forward but transposes internally
        model = model_class(input_dim, seq_len)
    else:
        model = model_class(input_dim, seq_len)
        
    model = model.to(device)
    model.train()
    
    # Generate random training batch
    x_batch = torch.randn(batch_size, seq_len, input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Warmup
    for _ in range(warmup_epochs):
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, x_batch if model_class != GNNProxy else x_batch[:, -1, :])
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
    # Benchmark loop
    start_time = time.perf_counter()
    for _ in range(run_epochs):
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, x_batch if model_class != GNNProxy else x_batch[:, -1, :])
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
    else:
        peak_vram = 0.0
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    time_per_epoch = total_time / run_epochs # seconds per batch
    
    return time_per_epoch, peak_vram

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # We test two dimensions: D=59 (HAI/SWaT/BATADAL) and D=127 (WADI)
    dimensions = [59, 127]
    models = [
        ("LSTM-AE", LSTMAE),
        ("Anomaly Transformer Proxy", AnomalyTransformerProxy),
        ("Graph Neural Network Proxy", GNNProxy)
    ]
    seq_len = 60
    batch_size = 256
    
    print("-" * 80)
    print(f"{'Model':<30} | {'Sensors (D)':<12} | {'Epoch Time (s)':<15} | {'100k Steps (s)':<15} | {'VRAM (MB)':<10}")
    print("-" * 80)
    
    for model_name, model_class in models:
        for D in dimensions:
            try:
                # Time per batch of size 256
                batch_time, peak_vram = run_benchmark(
                    device, model_class, input_dim=D, seq_len=seq_len, batch_size=batch_size
                )
                
                # Extrapolate to 100,000 samples
                # Number of batches in 100,000 samples = 100,000 / 256 = 390.6 batches
                batches_per_100k = 100000 / batch_size
                time_per_100k_epoch = batch_time * batches_per_100k
                
                print(f"{model_name:<30} | {D:<12} | {batch_time*1000:>.2f} ms/batch | {time_per_100k_epoch:>.2f} s | {peak_vram:>.2f} MB")
            except Exception as e:
                print(f"{model_name:<30} | {D:<12} | FAILED: {str(e)}")
    print("-" * 80)

if __name__ == "__main__":
    main()
