import torch
import torch.nn as nn
import torchvision.models as models

class HybridDRModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(HybridDRModel, self).__init__()
        
        # 1. CNN Backbone (ResNet50)
        # We remove the last fc layer and the avgpool layer to get feature maps
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # ResNet50 outputs 2048 channels. 
        # Feature map size at this point is usually 2048 x 16 x 16 for 512x512 input
        self.feature_dim = 2048
        
        # 2. Projection to Transformer dimension
        self.embed_dim = 512
        self.conv_proj = nn.Conv2d(self.feature_dim, self.embed_dim, kernel_size=1)
        
        # 3. Transformer Encoder
        # We flatten the spatial dimensions (H*W) to be the sequence length
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. Classification Head
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x):
        # x: [Batch, 3, 512, 512]
        
        # Backbone features
        x = self.backbone(x) # [Batch, 2048, 16, 16]
        
        # Project to embedding dim
        x = self.conv_proj(x) # [Batch, 512, 16, 16]
        
        # Flatten for Transformer: [Batch, Embed_Dim, Seq_Len] -> [Batch, Seq_Len, Embed_Dim]
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute(0, 2, 1) # [Batch, 256, 512]
        
        # Transformer
        x = self.transformer(x) # [Batch, 256, 512]
        
        # Global Average Pooling over sequence
        # Permute back to [Batch, Embed_Dim, Seq_Len] for pooling
        x = x.permute(0, 2, 1) # [Batch, 512, 256]
        x = self.avg_pool(x).squeeze(-1) # [Batch, 512]
        
        # Classification
        x = self.fc(x) # [Batch, Num_Classes]
        
        return x
