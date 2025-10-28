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
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2, pool=True, dilation=1, use_residual=False) -> None:
        super(Block,self).__init__()
        self.pool = pool
        # Adjust padding for dilation to maintain output size
        if dilation > 1:
            padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.ln = ConvLayerNorm(out_channels)
        if self.pool:
            self.pool = nn.MaxPool1d(pool_size)
        # use_residual parameter is ignored in non-residual Block, but kept for compatibility

    def forward(self,x):
        x = self.conv(x)
        x = self.ln(x)
        x = torch.relu(x)
        if self.pool:
            x = self.pool(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2, pool=True, dilation=1, use_residual=True) -> None:
        super(ResidualBlock, self).__init__()
        self.pool_flag = pool
        self.use_residual = use_residual

        # Adjust padding for dilation to maintain output size
        if dilation > 1:
            padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.ln = ConvLayerNorm(out_channels)

        if self.pool_flag:
            self.pool = nn.MaxPool1d(pool_size)

        # Projection layer for residual connection when channels change
        self.projection = None
        if use_residual and in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv(x)
        out = self.ln(out)
        out = torch.relu(out)

        # Residual connection (before pooling!)
        if self.use_residual:
            if self.projection is not None:
                identity = self.projection(identity)
            out = out + identity  # Skip connection

        # Pooling after residual (important!)
        if self.pool_flag:
            out = self.pool(out)

        return out
class TestModel(nn.Module):
    def __init__(self, dropout=0.5, use_dilation=False, base_channels=8, num_blocks=4, use_residual=True, return_features=False):
        super(TestModel, self).__init__()
        self.blocks = []
        self.return_features = return_features

        # Generate dilation pattern based on num_blocks
        if use_dilation:
            dilations = [2**i for i in range(num_blocks)]  # [1, 2, 4, 8, 16, ...]
        else:
            dilations = [1] * num_blocks

        # Choose block type based on use_residual
        BlockType = ResidualBlock if use_residual else Block

        # First block: 6 input channels -> base_channels
        self.blocks.append(BlockType(6, base_channels, dilation=dilations[0], use_residual=use_residual))

        # Middle blocks: base_channels -> base_channels (pool every other block)
        for i in range(1, num_blocks - 1):
            pool = (i % 2 == 1)  # Pool on odd indices (1, 3, 5...)
            self.blocks.append(BlockType(base_channels, base_channels,
                                        pool=pool, dilation=dilations[i], use_residual=use_residual))

        # Last block: base_channels -> base_channels*2 (no pool)
        self.blocks.append(BlockType(base_channels, base_channels * 2,
                                    pool=False, dilation=dilations[-1], use_residual=use_residual))

        self.blocks = nn.ModuleList(self.blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(base_channels * 2, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.gap(x).squeeze(-1)
        if self.return_features:
            return x
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
from torch.nn.functional import relu
class ObnoxiouslySimpleCNN(nn.Module):
    def __init__(self, input_channels, channels=[64,64], kernel_sizes=[7,5], dilations=[2,4], dropout=.5):
        super(ObnoxiouslySimpleCNN, self).__init__()
        self.stem = nn.Conv1d(input_channels, channels[0], kernel_size=7, padding=1, dilation=dilations[0]) # 3000 -> 2994, receptive field 7
        convs = []

        for in_channels, out_channels, kernel_size, dilation in zip(channels[:-1], channels[1:], kernel_sizes[1:], dilations[1:]):
            convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=((kernel_size - 1) * dilation) // 2,
                    dilation=dilation
                )
            )

        self.convs = nn.ModuleList(convs)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = self.stem(x)
        x = relu(x)
        for conv in self.convs:
            x = conv(x)
            x = relu(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)        
        return x