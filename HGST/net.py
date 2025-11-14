import torch.nn as nn
import torch
from function import normal
from function import calc_mean_std, normal
import scipy.stats as stats
from torchvision.utils import save_image
import torchvision
import numpy as np
import torch.nn.functional as F

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)
mlp = nn.ModuleList([nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)]) 
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out + 1e-7)  # 防止数值不稳定
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接池化结果
        x = self.conv1(x)
        x = self.sigmoid(x + 1e-7)  # 防止数值不稳定
        return x



        
class CCPL(nn.Module):
    def __init__(self, mlp):
        super(CCPL, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp
        self.channel_attention = ChannelAttention(in_planes=512)  # 假设输入通道是512
        self.spatial_attention = SpatialAttention(kernel_size=7)  # 7x7卷积

        self.channel_attention2 = ChannelAttention(in_planes=256)

    # def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
    #     b, c, h, w = feat.size()
    #     feat_r = feat.permute(0, 2, 3, 1).flatten(1, 2)
    #     if sample_ids == []:
    #         dic = {0: -(w+1), 1: -w, 2: -(w-1), 3: -1, 4: 1, 5: w-1, 6: w, 7: w+1}
    #         s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device) # indices of top left vectors
    #         s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
    #         ch_ids = (s_ids // (w - 2) + 1) # centors
    #         cw_ids = (s_ids % (w - 2) + 1)
    #         c_ids = (ch_ids * w + cw_ids).repeat(8)
    #         delta = [dic[i // num_s] for i in range(8 * num_s)]
    #         delta = torch.tensor(delta).to(feat.device)
    #         n_ids = c_ids + delta
    #         sample_ids += [c_ids]
    #         sample_ids += [n_ids]
    #     else:
    #         c_ids = sample_ids[0]
    #         n_ids = sample_ids[1]
    #     feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
    #     feat_d = feat_c - feat_n
    #     for i in range(3):
    #         feat_d =self.mlp[3*layer+i](feat_d)
    #     feat_d = Normalize(2)(feat_d.permute(0,2,1))
    #     return feat_d, sample_ids
    def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
        b, c, h, w = feat.size()
        if layer != 2:
            feat_r = feat * self.spatial_attention(feat)
            feat_r = feat_r * self.channel_attention(feat_r) 
        else:
            feat_r = feat * self.spatial_attention(feat)
            feat_r = feat_r * self.channel_attention2(feat_r) 
        feat_r = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if sample_ids == []:
            dic = {0: -(w+1), 1: -w, 2: -(w-1), 3: -1, 4: 1, 5: w-1, 6: w, 7: w+1}
            s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device) # indices of top left vectors
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
            ch_ids = (s_ids // (w - 2) + 1) # centors
            cw_ids = (s_ids % (w - 2) + 1)
            c_ids = (ch_ids * w + cw_ids).repeat(8)
            #delta = [dic[i // num_s] for i in range(8 * num_s)]
            delta = [dic[i // num_s] + torch.randint(low=-1, high=2, size=(1,)).item() for i in range(8 * num_s)]
            delta = torch.tensor(delta).to(feat.device)
            n_ids = c_ids + delta
            n_ids = n_ids.clamp(min=0, max=(h * w - 1))
            sample_ids += [c_ids]
            sample_ids += [n_ids]
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n
        for i in range(3):
            feat_d =self.mlp[3*layer+i](feat_d)
        feat_d = Normalize(2)(feat_d.permute(0,2,1))
        return feat_d, sample_ids



    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation 
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        ###
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return self.cross_entropy_loss(predictions, targets)
    def forward(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.07):
        loss_ccp = 0.0
        for i in range(start_layer, end_layer):
            f_q, sample_ids = self.NeighborSample(feats_q[i], i, num_s, [])
            f_k, _ = self.NeighborSample(feats_k[i], i, num_s, sample_ids)   
            loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
        return loss_ccp  

    # def forward(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.07):
    #     loss_ccp = 0.0
    #     for i in range(start_layer, end_layer):
    #         if i == 2:
    #             # 1. 将 relu_4 上采样到 relu_2 的空间维度
    #             relu_4_resized = F.interpolate(feats_q[4], size=(feats_q[2].size(2), feats_q[2].size(3)), mode='bilinear', align_corners=False).cuda()
                
    #             # 2. 调整 relu_4 的通道数，使其与 relu_2 的通道数一致
    #             conv_adjust_relu_4 = torch.nn.Conv2d(relu_4_resized.size(1), feats_q[2].size(1), kernel_size=1).cuda()
    #             relu_4_resized_adjusted = conv_adjust_relu_4(relu_4_resized)
                
    #             # 3. 融合两个特征图
    #             # 这里我们采用加法融合，也可以使用加权融合（例如乘上权重）
    #             fused_feature = relu_4_resized_adjusted + feats_q[2]  # 直接加和
                
    #             # 4. 使用卷积进一步融合特征，保持原始维度
    #             conv_fuse = torch.nn.Conv2d(fused_feature.size(1), feats_q[2].size(1), kernel_size=3, padding=1).cuda() 
    #             fused_feature = conv_fuse(fused_feature)  # 对融合后的特征进行卷积

    #             # 对 feats_k 进行相同的操作
    #             relu_4_resized_2 = F.interpolate(feats_k[4], size=(feats_k[2].size(2), feats_k[2].size(3)), mode='bilinear', align_corners=False).cuda()
    #             relu_4_resized_adjusted_2 = conv_adjust_relu_4(relu_4_resized_2)
    #             fused_feature_2 = relu_4_resized_adjusted_2 + feats_k[2]
    #             conv_fuse_2 = torch.nn.Conv2d(fused_feature_2.size(1), feats_k[2].size(1), kernel_size=3, padding=1).cuda() 
    #             fused_feature_2 = conv_fuse_2(fused_feature_2)
                
    #             # 5. 进行近邻采样
    #             f_q, sample_ids = self.NeighborSample(fused_feature, i, num_s, [])
    #             f_k, _ = self.NeighborSample(fused_feature_2, i, num_s, sample_ids) 
    #             loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
    #         else:
    #             f_q, sample_ids = self.NeighborSample(feats_q[i], i, num_s, [])
    #             f_k, _ = self.NeighborSample(feats_k[i], i, num_s, sample_ids)   
    #             loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
    #     return loss_ccp




class MCCNet(nn.Module):
    def __init__(self, in_dim):
        super(MCCNet, self).__init__()
        self.f = nn.Conv2d(in_dim , int(in_dim ), (1,1))
        self.g = nn.Conv2d(in_dim , int(in_dim ) , (1,1))
        self.h = nn.Conv2d(in_dim, int(in_dim), (1, 1))
        self.softmax  = nn.Softmax(dim=-2)
        self.out_conv = nn.Conv2d(int(in_dim ), in_dim, (1, 1))
        self.fc = nn.Linear(in_dim, in_dim)
    def forward(self,content_feat,style_feat):
        F_Fc_norm  = self.f(normal(content_feat))
        B,C,H,W = style_feat.size()

        G_Fs_norm =  self.g(normal(style_feat)).view(-1,1,H*W)
        G_Fs_sum = G_Fs_norm.view(B,C,H*W).sum(-1)

        FC_S = torch.bmm(G_Fs_norm,G_Fs_norm.permute(0,2,1)).view(B,C) /G_Fs_sum  #14
        FC_S = self.fc(FC_S).view(B,C,1,1)
        
        out = F_Fc_norm*FC_S
        B,C,H,W = content_feat.size()
        out = out.contiguous().view(B,-1,H,W)
        out = self.out_conv(out)
        out = content_feat + out
        return out

class MCC_Module(nn.Module):
    def __init__(self, in_dim):
        super(MCC_Module, self).__init__()
        self.MCCN=MCCNet(in_dim)

    def forward(self, content_feats, style_feats):
        content_feat_4 = content_feats[-2]
        style_feat_4 = style_feats[-2]
        Fcsc = self.MCCN(content_feat_4, style_feat_4)
       
        return Fcsc

class Net(nn.Module):
    def __init__(self, vgg, decoder ):
        super(Net, self).__init__()
        self.vgg=vgg
        #风格图feature提取
        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        self.mlp = mlp
        self.CCPL = CCPL(self.mlp)

        #transform
        self.mcc_module = MCC_Module(512)
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      #assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_ss_loss(self, fs):
        avg = nn.AvgPool2d(3,1)
        b, c, h, w = fs.shape
        x = avg(fs).view(b, c, -1, 1)
        y = avg(fs).view(b, c, 1, -1)
        x_norm = torch.norm(x)
        y_norm = torch.norm(y)
        dist = torch.matmul(x, y) / (x_norm * y_norm)
        cos_similarity = 1 - dist

        b, c, w, h = cos_similarity.shape

        min_n = torch.sum(torch.min(cos_similarity,dim=3)[0], dim=2) / h
        cos_similarity = cos_similarity.permute(0,1,3,2)
        min_m = torch.sum(torch.min(cos_similarity,dim=3)[0], dim=2) / w

        return torch.max(min_n,min_m,)

        return min

    def forward(self, content, style, tau=0.07, num_s=16, num_layer=3):
        s = torch.empty(1)
        t = torch.empty(content.size())

        std = torch.nn.init.uniform_(s, a=0.01, b=0.02)
        noise = torch.nn.init.normal_(t, mean=0, std=std[0]).cuda()
        content_noise = content + noise

        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        content_feats_N = self.encode_with_intermediate(content_noise)

        Ics = self.decoder(self.mcc_module(content_feats, style_feats))
        Ics_feats = self.encode_with_intermediate(Ics)

        Iins = self.calc_ss_loss(Ics_feats[-1])

        # Content loss
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))  +self.calc_content_loss(normal(Ics_feats[-3]), normal(content_feats[-3]))
        # Style loss
        loss_s = self.calc_style_loss(Ics_feats[-1], style_feats[-1])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

        # total variation loss
        y = Ics
        tv_loss = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

        Ics_N = self.decoder(self.mcc_module(content_feats_N, style_feats))
        loss_noise = self.calc_content_loss(Ics_N,Ics)

        #Identity losses lambda 1
        Icc = self.decoder(self.mcc_module(content_feats, content_feats))
        Iss = self.decoder(self.mcc_module(style_feats, style_feats))
        loss_lambda1 = self.calc_content_loss(Icc,content)+self.calc_content_loss(Iss,style)
        
        #Identity losses lambda 2
        Icc_feats=self.encode_with_intermediate(Icc)
        Iss_feats=self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0])+self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i])+self.calc_content_loss(Iss_feats[i], style_feats[i])

        start_layer = 5 - num_layer
        loss_ccp = self.CCPL(Ics_feats, content_feats, num_s, start_layer, 4)
        return loss_noise, loss_c, loss_s, loss_lambda1, loss_lambda2,tv_loss, Ics, Iins, loss_ccp
        #return loss_c, loss_s,loss_lambda1, loss_lambda2, tv_loss