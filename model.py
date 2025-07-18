import numpy as np
import pickle
import torch
import torch.nn as nn
import layer as l
from torch.nn import functional as F, Parameter
import pickle
import torch


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = args.device

    @staticmethod
    def format_metrics(metrics, split):
        return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

    @staticmethod
    def has_improved(m1, m2):
        return (m1["Mean Rank"] > m2["Mean Rank"]) or (m1["Mean Reciprocal Rank"] < m2["Mean Reciprocal Rank"])

    @staticmethod
    def init_metric_dict():
        return {"Hits@100": -1, "Hits@10": -1, "Hits@3": -1, "Hits@1": -1,
                "Mean Rank": 100000, "Mean Reciprocal Rank": -1}


class DiffusionModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.denoising_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.noise_strength = 0.1

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            return self.reverse_diffusion(x)
        else:
            return self.forward_diffusion(x)
    def forward_diffusion(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.noise_strength
        return x + noise
    def reverse_diffusion(self, x: torch.Tensor) -> torch.Tensor:
        return self.denoising_net(x)

class MEOW(BaseModel):
    def __init__(self, args):
        super(MEOW, self).__init__(args)

        self.entity_diffusion = DiffusionModel(args.dim)
        self.relation_diffusion = DiffusionModel(args.r_dim)

        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('dataset/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(),
                freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('dataset/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(
                    pickle.load(open('dataset/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0),
                freeze=False)
        img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(8, 64))
        img = img_pool(args.img.to(self.device).view(-1, 64, 64))
        img = img.view(img.size(0), -1)
        '''img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(args.img.to(self.device).view(-1, 64, 64))
        img = img.view(img.size(0), -1)'''
        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)

        txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(8, 64))
        txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
        txt = txt.view(txt.size(0), -1)
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        self.dim = args.dim
        self.TuckER_S = l.TuckERLayer(args.dim, args.r_dim)
        self.TuckER_I = l.TuckERLayer(args.dim, args.r_dim)
        self.TuckER_D = l.TuckERLayer(args.dim, args.r_dim)
        self.TuckER_MM = l.TuckERLayer(args.dim, args.r_dim)
        self.Mutan_MM_E = l.MutanLayer_kl(args.dim, 2)
        self.Mutan_MM_R = l.MutanLayer_kl(args.dim, 2)
        # self.Mutan_MM_E = l.MutanLayer_JS(args.dim, 2)
        # self.Mutan_MM_R = l.MutanLayer_JS(args.dim, 2)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.bceloss = nn.BCELoss()
        self.criterion = nn.MSELoss(reduction='mean')
        self.decoder1 = nn.Sequential(
            torch.nn.Linear(512, 512),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            torch.nn.Linear(512, 512),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            torch.nn.Linear(512, 512),
            nn.ReLU()
        )
        self.register_parameter('b', Parameter(torch.zeros(len(args.entity2id))))

    def contrastive_loss(self, s_embed, v_embed, t_embed):
        s_embed, v_embed, t_embed = s_embed / torch.norm(s_embed), v_embed / torch.norm(v_embed), t_embed / torch.norm(
            t_embed)
        pos_sv = torch.sum(s_embed * v_embed, dim=1, keepdim=True)
        pos_st = torch.sum(s_embed * t_embed, dim=1, keepdim=True)
        pos_vt = torch.sum(v_embed * t_embed, dim=1, keepdim=True)
        neg_s = torch.matmul(s_embed, s_embed.t())
        neg_v = torch.matmul(v_embed, v_embed.t())
        neg_t = torch.matmul(t_embed, t_embed.t())
        neg_s = neg_s - torch.diag_embed(torch.diag(neg_s))
        neg_v = neg_v - torch.diag_embed(torch.diag(neg_v))
        neg_t = neg_t - torch.diag_embed(torch.diag(neg_t))
        pos = torch.mean(torch.cat([pos_sv, pos_st, pos_vt], dim=1), dim=1)
        neg = torch.mean(torch.cat([neg_s, neg_v, neg_t], dim=1), dim=1)
        loss = torch.mean(F.softplus(neg - pos))
        return loss

    def forward(self, batch_inputs, train_mode=True):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        # e_embed = self.fc(e_embed)
        # r_embed = self.fc(r_embed)
        e_img_embed = self.img_entity_embeddings(head)
        r_img_embed = self.img_relation_embeddings(relation)
        e_txt_embed = self.txt_entity_embeddings(head)
        r_txt_embed = self.txt_relation_embeddings(relation)
        e_mm_output = self.Mutan_MM_E(e_embed, e_img_embed, e_txt_embed)
        r_mm_output = self.Mutan_MM_R(r_embed, r_img_embed, r_txt_embed)
        e_mm_embed = e_mm_output[0]
        r_mm_embed = r_mm_output[0]
        # e_skl_loss = e_mm_output[1]
        e_js_loss = e_mm_output[1]
        # r_skl_loss = r_mm_embed[1]
        e_model1 = self.decoder1(e_mm_embed)
        e_model2 = self.decoder2(e_mm_embed)
        e_model3 = self.decoder3(e_mm_embed)
        # r_model1, r_model2, r_model3 = r_mm_embed[2:]
        recon_error = self.criterion(e_embed, e_model1) + self.criterion(e_img_embed, e_model2) + self.criterion(
            e_txt_embed, e_model3)
        # recon_error = self.bceloss(e_embed, e_model1) + self.bceloss(e_img_embed, e_model2) + self.bceloss(e_txt_embed, e_model3)
        if train_mode:
            noisy_e_embed = self.entity_diffusion(e_embed, reverse=False)
            noisy_e_img_embed = self.entity_diffusion(e_img_embed, reverse=False)
            noisy_e_txt_embed = self.entity_diffusion(e_txt_embed, reverse=False)
            noisy_e_mm_embed = self.entity_diffusion(e_mm_embed, reverse=False)
            #noisy_r_embed = self.relation_diffusion(r_mm_embed, reverse=False)
        else:
            noisy_e_embed = e_embed
            noisy_e_img_embed = e_img_embed
            noisy_e_txt_embed = e_txt_embed
            noisy_e_mm_embed = e_mm_embed
            #noisy_r_embed = r_mm_embed

            # 逆向扩散：尝试恢复原始数据
        recovered_e_embed = self.entity_diffusion(noisy_e_embed, reverse=True)
        recovered_e_img_embed = self.entity_diffusion(noisy_e_img_embed, reverse=True)
        recovered_e_txt_embed = self.entity_diffusion(noisy_e_txt_embed, reverse=True)
        recovered_e_mm_embed = self.entity_diffusion(noisy_e_mm_embed, reverse=True)
        #recovered_r_embed = self.relation_diffusion(noisy_r_embed, reverse=True)
        #pred_s = self.TuckER_S(recovered_e_embed, r_embed)
        pred_i = self.TuckER_I(recovered_e_img_embed, r_img_embed)
        #pred_d = self.TuckER_D(recovered_e_txt_embed, r_txt_embed)

        #pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)

        pred_s = self.TuckER_S(e_embed, r_embed)
        #pred_i = self.TuckER_I(e_img_embed, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        #pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)
        pred_mm = self.TuckER_MM(recovered_e_mm_embed, r_mm_embed)

        pred_s = F.relu(pred_s)
        pred_i = F.relu(pred_i)
        pred_d = F.relu(pred_d)
        pred_mm = F.relu(pred_mm)

        pred_s = torch.mm(pred_s, self.entity_embeddings.weight.transpose(1, 0))
        pred_i = torch.mm(pred_i, self.img_entity_embeddings.weight.transpose(1, 0))
        pred_d = torch.mm(pred_d, self.txt_entity_embeddings.weight.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, self.Mutan_MM_E(self.entity_embeddings.weight,
                                                    self.img_entity_embeddings.weight,
                                                    self.txt_entity_embeddings.weight)[0].transpose(1, 0))

        pred_s += self.b.expand_as(pred_s)
        pred_i += self.b.expand_as(pred_i)
        pred_d += self.b.expand_as(pred_d)
        pred_mm += self.b.expand_as(pred_mm)

        pred_s = torch.sigmoid(pred_s)
        pred_i = torch.sigmoid(pred_i)
        pred_d = torch.sigmoid(pred_d)
        pred_mm = torch.sigmoid(pred_mm)




        return [pred_s, pred_i, pred_d, pred_mm, e_embed, e_img_embed, e_txt_embed, e_mm_embed, e_js_loss,recon_error,recovered_e_img_embed, recovered_e_mm_embed,recovered_e_embed,recovered_e_txt_embed]#,recovered_e_embed

    def loss_func(self, output, target,weight):
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)
        recovery_loss_m = self.criterion(output[7], output[11].detach())  # e_embed vs recovered_e_embed
        recovery_loss_s = self.criterion(output[4], output[12].detach()) # r_embed vs recovered_r_embed
        recovery_loss_img = self.criterion(output[5], output[10].detach())
        recovery_loss_txt = self.criterion(output[6], output[13].detach())


        loss_cl = self.contrastive_loss(output[4], output[5], output[6])  # s_embed, v_embed, t_embed

        total_loss = loss_s + loss_i + loss_d + loss_mm + weight*loss_cl + weight*output[8] + weight*output[9] + recovery_loss_img + recovery_loss_m + recovery_loss_s+recovery_loss_txt
        return total_loss




