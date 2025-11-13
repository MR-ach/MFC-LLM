import os
import torch
from torch import nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        self.embed_dim = in_channels
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Conv1d(in_channels, in_channels, 1)
        self.norm = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
    def forward(self, query, key, value):
        B, C, L = query.shape
        q = query.permute(0, 2, 1)
        k = key.permute(0, 2, 1)
        v = value.permute(0, 2, 1)
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.permute(0, 2, 1)
        attn_out = self.proj(attn_out)
        attn_out = self.norm(attn_out)
        attn_out = self.relu(attn_out)
        return attn_out
    
class ConvWide(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=8):
        super(ConvWide, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.ca = ChannelAttention(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ConvMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvMultiScale, self).__init__()
        if out_channels % 4 != 0:
            raise ValueError('out_channels should be divisible by 4')
        out_channels = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, 4, padding=0)
        self.conv3 = nn.Conv1d(in_channels, out_channels, 3, 4, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, 5, 4, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, 7, 4, padding=3)
        self.norm = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(out_channels * 3)
    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.norm(x)
        x = self.relu(x)
        x = self.ca(x) * x
        x = torch.cat([x1, x], dim=1)
        return x
    
class ChannelAttention(nn.Module): 
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.se(self.avg_pool(x))
        max_out = self.se(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
   
class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.conv_query = ConvWide(1, 60, 8, 8)
        self.conv_ref = ConvWide(1, 8, 8, 8)
        self.conv_res = ConvWide(1, 60, 8, 8)
        self.align_query = nn.Conv1d(60, 64, 1)
        self.align_ref = nn.Conv1d(8, 32, 1)
        self.align_res = nn.Conv1d(60, 32, 1)
        self.cross_attn = CrossAttentionBlock(in_channels=128, num_heads=4)
        self.conv = nn.Sequential(
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128)
        )
    def forward(self, x):
        query = x[:, :1, :]
        ref = x[:, 1:2, :]
        res = query - ref
        query_feat = self.conv_query(query)   
        ref_feat = self.conv_ref(ref)        
        res_feat = self.conv_res(res)         
        query_feat_aligned = self.align_query(query_feat)   
        ref_feat_aligned = self.align_ref(ref_feat)         
        res_feat_aligned = self.align_res(res_feat)        
        feat = torch.cat([query_feat_aligned, ref_feat_aligned, res_feat_aligned], dim=1) 
        attn_out = self.cross_attn(feat, feat, feat)  
        feat = feat + attn_out  
        feat = self.conv(feat)
        return feat    
    def save_weights(self, weights_dir):
        torch.save(self.state_dict(), weights_dir + '/feature_encoder.pth')
    def load_weights(self, weights_dir):
        self.load_state_dict(torch.load(weights_dir + '/feature_encoder.pth', map_location='cpu'))

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(1024, 128) 
        self.linear2 = nn.Linear(128, 5)  
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)
    def save_weights(self, weights_dir):
        torch.save(self.state_dict(), weights_dir + '/classifier.pth')
    def load_weights(self, weights_dir):
        self.load_state_dict(torch.load(weights_dir + '/classifier.pth', map_location='cpu'))

class LifeRegressor(nn.Module):
    def __init__(self):
        super(LifeRegressor, self).__init__()
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, 1)  
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)
    def save_weights(self, weights_dir):
        torch.save(self.state_dict(), os.path.join(weights_dir, 'regressor.pth'))
    def load_weights(self, weights_dir):
        self.load_state_dict(torch.load(os.path.join(weights_dir, 'regressor.pth'), map_location='cpu'))

class F2LNet(nn.Module):
    def __init__(self):
        super(F2LNet, self).__init__()
        self.encoder = FeatureEncoder()
        self.classifier = Classifier()
        self.regressor = LifeRegressor()
    def forward(self, x):
        feat = self.encoder(x)
        cls_out = self.classifier(feat)
        life_out = self.regressor(feat)
        return cls_out, life_out
    def save_weights(self, weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
        self.encoder.save_weights(weights_dir)
        self.classifier.save_weights(weights_dir)
        self.regressor.save_weights(weights_dir)
    def load_weights(self, weights_dir):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(f'{weights_dir} not found')
        self.encoder.load_weights(weights_dir)
        self.classifier.load_weights(weights_dir)
        self.regressor.load_weights(weights_dir)

if __name__ == '__main__':
    def test():
        test_signal = torch.randn(8, 3, 4096)  
        model = F2LNet()
        cls_out, life_out = model(test_signal)
        print("分类输出形状:", cls_out.shape)   
        print("寿命输出形状:", life_out.shape)  
        model.save_weights('test_multi')
        model.load_weights('test_multi')
        os.remove('test_multi/feature_encoder.pth')
        os.remove('test_multi/classifier.pth')
        os.remove('test_multi/regressor.pth')
        os.rmdir('test_multi')
    test()
