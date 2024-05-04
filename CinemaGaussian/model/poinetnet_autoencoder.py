import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Autoencoder based on pointnet 
'''

# Spatial Transformer Networks
class STNet(nn.Module):
    def __init__(self, k):
        super(STNet, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, k * k)
        self.activation = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)


    def forward(self, pc):
        B, N, D = pc.size()                 #[B, N, D]
        pc = pc.permute(0,2,1)               #[B, D, N]
        pc = self.activation(self.bn1(self.conv1(pc)))    #[B, 64, N]
        pc = self.activation(self.bn2(self.conv2(pc)))    #[B, 128, N]
        pc = self.activation(self.bn3(self.conv3(pc)))    #[B, 1024, N]
        
        pc = torch.max(pc, dim=2, keepdim=False)[0]            #[B, 1024]
    
        pc = self.activation(self.linear1(pc))          #[B, 512]
        pc = self.activation(self.linear2(pc))          #[B, 256]
        pc = self.linear3(pc).view((B, self.k, self.k)) #[B, k , k]

        iden = torch.eye(self.k).repeat(B, 1, 1).to(pc.device)
        pc = pc + iden   # B * k * k
        return pc


class GaussianEncoder(nn.Module):
    def __init__(self, input_channels, useSTN = True):
        super(GaussianEncoder, self).__init__()
        self.useSTN = useSTN
        self.stn = STNet(k=3)           #Position transform
        self.conv1 = torch.nn.Conv1d(input_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.fstn = STNet(k=64)        #Feature transform
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.activation = nn.ReLU()
        
    def forward(self, pc):
        B, N, D = pc.size()

        # position transform
        if self.useSTN:
            pos = pc[:,:,:3]
            feature = pc[:,:,3:]
            trans = self.stn(pos)
            pos = torch.bmm(pos, trans)
            pc = torch.cat([pos, feature], dim=2)   #[B, N, D]
        
        
        pc = pc.permute(0,2,1)                  #[B, D, N]

        fea1 = self.activation(self.bn1(self.conv1(pc)))     #[B, 64, N]
        fea2 = self.activation(self.bn2(self.conv2(fea1)))   #[B, 64, N]
        
        # feature transform
        if self.useSTN:
            trans_mat = self.fstn(fea2.permute(0,2,1))
            trans_fea = torch.bmm(fea2.transpose(2, 1), trans_mat)  #[B, N, 64]
            trans_fea = trans_fea.permute(0, 2, 1)
        else:
            trans_fea = fea2

        
        fea3 = self.activation(self.bn3(self.conv3(trans_fea)))   #[B, 128, N]
        fea4 = self.bn4(self.conv4(fea3))                           #[B, 1024, N]
        global_feature = torch.max(fea4, dim=2)[0].view(-1, 1024)   #[B, 1024]

        expand = global_feature.view(-1, 1024, 1).repeat(1, 1, N)     #[B, 1024, N]
        #1024 + 64+64+128
        point_feature = torch.cat([expand, fea1, fea2, fea3], dim=1)   #[B, 1280, N]

        return point_feature

class GaussianDecoder(nn.Module):
    def __init__(self, input_channels):
        super(GaussianDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(1280, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, input_channels, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.activation = nn.ReLU()
        
        
    def forward(self, point_fea):
        # point_fea [B, 1280, N]
        fea = self.activation(self.bn1(self.conv1(point_fea)))     #[B, 1024, N]
        fea = self.activation(self.bn2(self.conv2(fea)))           #[B, 128, N]
        fea = self.activation(self.bn3(self.conv3(fea)))           #[B, 64, N]
        fea = self.conv4(fea)                                       #[B, D, N]
        return fea

class GaussianAutoEncoder(nn.Module):
    def __init__(self, input_channels, useSTN = True):
        super(GaussianAutoEncoder, self).__init__()
        self.encoder = GaussianEncoder(input_channels=input_channels, useSTN = useSTN)
        self.decoder = GaussianDecoder(input_channels=input_channels)

    def forward(self, pc):
        N, D = pc.size()            #[N, D]
        pc = pc.unsqueeze(dim=0)    # add batch dim [1, N, D]
        point_feature = self.encoder(pc)
        point_data = self.decoder(point_feature)
        point_data = point_data.permute(0,2,1).reshape(N, D) #[N, D]
        point_feature = point_feature.permute(0,2,1).reshape(N, 1280) #[N, 1280]
        return point_data, point_feature
    
    def get_globalfea(self, clusters):
        lens = [clu.shape[0] for clu in clusters]
        points = torch.vstack(clusters)
        points = torch.cat([points[:,:6], points[:,-8:]], dim=1).unsqueeze(dim=0)
        points_feature = self.encoder(points).squeeze().transpose(0,1)
        clu_features = torch.split(points_feature, lens, dim=0)
        clu_global_feature = [torch.max(clu, dim=0)[0] for clu in clu_features]
        return torch.vstack(clu_global_feature)
