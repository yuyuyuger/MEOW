import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
CUDA = torch.cuda.is_available()




def kl_divergence(mu1, sigma1, mu2, sigma2):
    kl = 0.5 * (2 * torch.log(sigma2) - 2 * torch.log(sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / sigma2 ** 2 - 1)
    return kl.sum(dim=-1)

class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(512, 440),
            nn.ReLU(True),
            nn.Linear(440, z_dim * 2),
        )


    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = F.softplus(sigma) + 1e-6
        return mu, sigma


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        #self.encoder3 = Encoder()

    def forward(self, x_modal1, x_modal2):

        mu1, sigma1 = self.encoder1(x_modal1)
        mu2, sigma2 = self.encoder2(x_modal2)
        #mu3, sigma3 = self.encoder3(x_modal3)
        # 计算KL散度损失
        kl_div = kl_divergence(mu1, sigma1, mu2, sigma2)  # 示例中的函数需要根据实际情况替换
        #kl_div += kl_divergence(mu2, sigma2, mu3, sigma3)
        #kl_div += kl_divergence(mu3, sigma3, mu1, sigma1)

        loss = kl_div.mean()

        return loss

def js_divergence(mu1, sigma1, mu2, sigma2):
    # 计算M，两个分布的平均
    mu_m = 0.5 * (mu1 + mu2)
    sigma_m = 0.5 * (sigma1 + sigma2)

    # 计算KL散度
    kl1 = kl_divergence(mu1, sigma1, mu_m, sigma_m)
    kl2 = kl_divergence(mu2, sigma2, mu_m, sigma_m)

    # 计算JS散度
    js = 0.5 * (kl1 + kl2)
    return js


class AmbiguityLearning_JS(nn.Module):
    def __init__(self):
        super(AmbiguityLearning_JS, self).__init__()
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        #self.encoder3 = Encoder()

    def forward(self, x_modal1, x_modal2):
        mu1, sigma1 = self.encoder1(x_modal1)
        mu2, sigma2 = self.encoder2(x_modal2)
        #mu3, sigma3 = self.encoder3(x_modal3)

        # 计算JS散度损失
        js_div = js_divergence(mu1, sigma1, mu2, sigma2)
        #js_div += js_divergence(mu2, sigma2, mu3, sigma3)
        #js_div += js_divergence(mu3, sigma3, mu1, sigma1)

        loss = js_div.mean()
        return loss


class MutanLayer(nn.Module):
    def __init__(self, dim, multi):
        super(MutanLayer, self).__init__()

        self.dim = dim
        self.multi = multi

        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        bs = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_mm.append(torch.mul(torch.mul(x_modal1, x_modal2), x_modal3))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)
        return x_mm

class MutanLayer_kl(nn.Module):
    def __init__(self, dim, multi):
        super(MutanLayer_kl, self).__init__()

        self.dim = dim
        self.multi = multi
        self.kl_divergence = AmbiguityLearning()  # 将 AmbiguityLearning 类实例化为属性

        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.3)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.3)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.3)
            lin = nn.Linear(dim, dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)



    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        bs = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_mm.append(torch.mul(torch.mul(x_modal1, x_modal2), x_modal3))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)

        # 计算 KL 散度损失
        kl_loss = self.kl_divergence(x_modal2, x_modal3)

        # 解码单模态特征

        return [x_mm, kl_loss]

class MutanLayer_JS(nn.Module):
    def __init__(self, dim, multi):
        super(MutanLayer_JS, self).__init__()

        self.dim = dim
        self.multi = multi
        self.js_divergence = AmbiguityLearning_JS()  # 将 AmbiguityLearning 类实例化为属性

        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.3)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.3)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.3)
            lin = nn.Linear(dim, dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)



    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        bs = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_mm.append(torch.mul(torch.mul(x_modal1, x_modal2), x_modal3))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)

        # 计算 KL 散度损失
        js_loss = self.js_divergence(x_modal2, x_modal3)

        # 解码单模态特征

        return [x_mm, js_loss]

class MMencoder(nn.Module):
    def __init__(self):
        super(MMencoder, self).__init__()
        # 激活函数
        self.activation = nn.ReLU()
        self.fc_z = nn.Linear(512, 256)

        self.kl_divergence = AmbiguityLearning()
        self.conv1d = nn.Sequential(nn.Conv1d(768, 512, kernel_size=1), self.activation)



    def forward(self, e_embed, e_img_embed, e_txt_embed):


        self_att_t = self.activation(e_txt_embed) # 应该输出torch.Size([256, 512, 1])
        self_att_i = self.activation(e_img_embed)
        self_att_s = self.activation(e_embed)

        self_att_i_ = self.fc_z((self_att_i).view(-1, 512))
        self_att_t_ = self.fc_z((self_att_t).view(-1, 512))
        self_att_s_ = self.fc_z((self_att_s).view(-1, 512))

        x_mm = torch.cat([self_att_i_, self_att_s_], dim=1)
        x_mm = torch.cat([x_mm, self_att_t_], dim=1)

        x_mm = x_mm.unsqueeze(2)
        bsz = e_txt_embed.size()[0]
        x_mm = self.conv1d(x_mm).view(bsz, 512)
        kl_loss = self.kl_divergence(e_embed, e_img_embed, e_txt_embed)

        return [x_mm, kl_loss]
class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim // num_heads
        self.scale = self.dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3, batch_size, num_heads, seq_len, dim

        scores = torch.einsum('bhqd, bhkd -> bhqk', q, k) * self.scale
        attn = torch.softmax(scores, dim=-1)

        out = torch.einsum('bhqk, bhvd -> bhqd', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)

        return self.fc_out(out)

class MutanLayer_conv(nn.Module):
    def __init__(self, set_channels, part_dim, output_dim):
        super(MutanLayer_conv, self).__init__()

        self.set_channels = set_channels
        self.part_dim = part_dim
        self.output_dim = output_dim

        self.kl_divergence = AmbiguityLearning()  # 将 AmbiguityLearning 类实例化为属性
        self.fc_preprocess = nn.Linear(512, set_channels[0] * part_dim)

        # 定义多个阶段的卷积层，用于每个模态的特征提取
        self.conv_stage1 = BasicConv1d(set_channels[0], set_channels[1], kernel_size=3, padding=1)
        self.conv_stage2 = BasicConv1d(set_channels[1], set_channels[2], kernel_size=3, padding=1)

        # 定义融合层，融合不同模态的特征
        self.fusion_fc1 = nn.Linear(set_channels[1] * 3, set_channels[2])
        self.fusion_fc2 = nn.Linear(set_channels[2] * 3, set_channels[2])

        # 定义时空注意力机制
        self.attention = Attention(dim=set_channels[2] * 2)  # 因为拼接了两个阶段

        # 定义最终的融合全连接层
        self.final_fc = nn.Linear(set_channels[2] * 2, output_dim)

        # Visual and Text Enhancer


    def forward(self, structural, visual, textual):
        # 将输入通过全连接层进行预处理
        structural = self.fc_preprocess(structural).view(-1, self.set_channels[0], self.part_dim)
        visual = self.fc_preprocess(visual).view(-1, self.set_channels[0], self.part_dim)
        textual = self.fc_preprocess(textual).view(-1, self.set_channels[0], self.part_dim)

        # 第一个阶段的卷积处理
        structural1 = self.conv_stage1(structural)
        visual1 = self.conv_stage1(visual)
        textual1 = self.conv_stage1(textual)
        # 第一阶段融合
        fused1 = torch.cat([structural1, visual1, textual1], dim=1)
        fused1 = self.fusion_fc1(fused1.permute(0, 2, 1)).permute(0, 2, 1)

        # 第二个阶段的卷积处理
        structural2 = self.conv_stage2(structural1)
        visual2 = self.conv_stage2(visual1)
        textual2 = self.conv_stage2(textual1)
        # 第二阶段融合
        fused2 = torch.cat([structural2, visual2, textual2], dim=1)
        fused2 = self.fusion_fc2(fused2.permute(0, 2, 1)).permute(0, 2, 1)

        # 将第一阶段和第二阶段的融合结果拼接
        combined = torch.cat([fused1, fused2], dim=2)

        # 应用时空注意力机制
        combined = self.attention(combined.permute(0, 2, 1))

        # 最终的融合全连接层
        output = self.final_fc(combined.mean(dim=1))

        # 保留的部分
        kl_loss = self.kl_divergence(structural2, visual2, textual2)

        return [output, kl_loss]



class TuckERLayer(nn.Module):
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()
        
        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.3)
        self.out_drop = nn.Dropout(0.3)

    def forward(self, e_embed, r_embed):
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))
        
        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
        r = r.view(-1, x.size(2), x.size(2))
        r = self.hidden_drop(r)
       
        x = torch.bmm(x, r)
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.out_drop(x)
        return x


class ConvELayer(nn.Module):
    def __init__(self, dim, out_channels, kernel_size, k_h, k_w):
        super(ConvELayer, self).__init__()

        self.input_drop = nn.Dropout(0.2)
        self.conv_drop = nn.Dropout2d(0.2)
        self.hidden_drop = nn.Dropout(0.2)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm1d(dim)

        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                                    stride=1, padding=0, bias=True)
        assert k_h * k_w == dim
        flat_sz_h = int(2*k_w) - kernel_size + 1
        flat_sz_w = k_h - kernel_size + 1
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels
        self.fc = nn.Linear(self.flat_sz, dim, bias=True)

    def forward(self, conv_input):
        x = self.bn0(conv_input)
        x = self.input_drop(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices
            if(CUDA):
                edge_sources = edge_sources.to('cuda:0')
            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(out_features, 2 * in_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge):
        N = input.size()[0]

        edge_h = torch.cat((input[edge[0, :], :], input[edge[1, :], :]), dim=1).t()
        # edge_h: (2*in_dim) x E

        edge_m = self.W.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_m).squeeze())).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(edge, edge_w, N, edge_w.shape[0], self.out_features)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention
        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(nfeat, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nheads * nhid,
                                             nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, entity_embeddings, relation_embeddings, edge_list):
        x = entity_embeddings
        x = torch.cat([att(x, edge_list) for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x_rel = relation_embeddings.mm(self.W)
        x = F.elu(self.out_att(x, edge_list))
        return x, x_rel

