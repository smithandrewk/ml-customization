import torch
from torch import nn

class ConvLayerNorm(nn.Module):
    # might actually be instance norm haha jun19
    
    def __init__(self, out_channels) -> None:
        super(ConvLayerNorm,self).__init__()
        self.ln = nn.LayerNorm(out_channels, elementwise_affine=False)

    def forward(self,x):
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2, pool=True) -> None:
        super(Block,self).__init__()
        self.pool = pool
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.ln = ConvLayerNorm(out_channels)
        if self.pool:
            self.pool = nn.MaxPool1d(pool_size)

    def forward(self,x):
        x = self.conv(x)
        x = self.ln(x)
        x = torch.relu(x)
        if self.pool:
            x = self.pool(x)
        return x
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.blocks = []
        self.blocks.append(Block(6,8))
        for _ in range(1):
            self.blocks.append(Block(8,8))
            self.blocks.append(Block(8,8,pool=False))

        self.blocks.append(Block(8,16,pool=False))
            
        self.blocks = nn.ModuleList(self.blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x