import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .GRU import BIGRU

def loss_kld(inputs, targets):
    inputs = F.log_softmax(inputs, dim=1)
    targets = F.softmax(targets, dim=1)
    return F.kl_div(inputs, targets, reduction='batchmean')

# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    print('GRU weights initialization finished!')

def weights_init_embed(m):
    with torch.no_grad():
        for child in list(m.children()):
            classname = child.__class__.__name__
            if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
                nn.init.xavier_uniform_(list(child.parameters())[0])
                #child.weight.data.normal_(0.0, 0.02)
                #if child.bias is not None:
                #    child.bias.data.fill_(0)

class CSTCN_MoCo(nn.Module):
    def __init__(self, skeleton_representation, args_bi_gru, dim=128, K=65536, m=0.999, T=0.07,
                 teacher_T=0.05, student_T=0.1, cmd_weight=1.0, topk=1024, mlp=False, nmb_prototypes=128,
                 sk_epsilon=0.05, sk_T=0.1, batch_size=64, dataset='NTU60', sigm=2.0, pretrain=True):
        super(CSTCN_MoCo, self).__init__()
        self.pretrain = pretrain
        self.dataset = dataset
        self.batch_size = batch_size
        # self.nmb_prototypes = nmb_prototypes
        # self.skeleton_representation = skeleton_representation
        self.nmb_prototypes = 128  # spatial
        self.nmb_prototypes_T = 128  # temporal
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        if not self.pretrain:
            self.encoder_q = BIGRU(**args_bi_gru)
            self.encoder_q_motion = BIGRU(**args_bi_gru)
            self.encoder_q_bone = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
            weights_init_gru(self.encoder_q_motion)
            weights_init_gru(self.encoder_q_bone)
        else:
            self.K = K
            self.m = m
            self.T = T
            self.teacher_T = teacher_T
            self.student_T = student_T
            self.cmd_weight = cmd_weight
            self.topk = topk
            self.sk_epsilon = sk_epsilon
            self.sk_T = sk_T
           
            self.sigm = sigm
            mlp=mlp

            ###########prototype###########

            self.prototype_q = nn.Linear(128, self.nmb_prototypes, bias=False)

            self.prototype_q_motion = nn.Linear(128, self.nmb_prototypes, bias=False)

            self.prototype_q_bone = nn.Linear(128, self.nmb_prototypes, bias=False)

            torch.nn.init.xavier_uniform_(self.prototype_q.weight.data)
            torch.nn.init.xavier_uniform_(self.prototype_q_motion.weight.data)
            torch.nn.init.xavier_uniform_(self.prototype_q_bone.weight.data)

            self.prototype_q_sequece = nn.Linear(128, self.nmb_prototypes_T , bias=False)

            self.prototype_q_motion_sequece = nn.Linear(128, self.nmb_prototypes_T , bias=False)

            self.prototype_q_bone_sequece = nn.Linear(128, self.nmb_prototypes_T , bias=False)

            torch.nn.init.xavier_uniform_(self.prototype_q_sequece.weight.data)
            torch.nn.init.xavier_uniform_(self.prototype_q_motion_sequece.weight.data)
            torch.nn.init.xavier_uniform_(self.prototype_q_bone_sequece.weight.data)

            self.encoder_q = BIGRU(**args_bi_gru)
            self.encoder_k = BIGRU(**args_bi_gru)
            self.encoder_q_motion = BIGRU(**args_bi_gru)
            self.encoder_k_motion = BIGRU(**args_bi_gru)
            self.encoder_q_bone = BIGRU(**args_bi_gru)
            self.encoder_k_bone = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
            weights_init_gru(self.encoder_q_motion)
            weights_init_gru(self.encoder_q_bone)
            weights_init_gru(self.encoder_k)
            weights_init_gru(self.encoder_k_motion)
            weights_init_gru(self.encoder_k_bone)
            ######### Frame semantic
            frame = 64
            self.tem = self.one_hot(self.batch_size, frame, 25)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda() #(bs,64,25,64) 64 frame
            self.tem_embed = nn.Sequential(
                                            nn.Conv2d(frame, self.nmb_prototypes_T, kernel_size=1, bias=True),
                                            nn.ReLU(),
                                            nn.Conv2d(self.nmb_prototypes_T, self.nmb_prototypes_T, kernel_size=1, bias=True),
                                            nn.ReLU(),
                                            nn.AdaptiveMaxPool2d((1, frame)),
                                            )   # (bs,2048,25,64)
            weights_init_embed(self.tem_embed)


            #projection heads
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_k.fc)
                self.encoder_q_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                            nn.ReLU(),
                                                            self.encoder_q_motion.fc)
                self.encoder_k_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                            nn.ReLU(),
                                                            self.encoder_k_motion.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_q_bone.fc)
                self.encoder_k_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_k_bone.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient
            for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue", torch.randn(dim, self.K))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_motion", torch.randn(dim, self.K))
            self.queue_motion = F.normalize(self.queue_motion, dim=0)
            self.register_buffer("queue_ptr_motion", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_bone", torch.randn(dim, self.K))
            self.queue_bone = F.normalize(self.queue_bone, dim=0)
            self.register_buffer("queue_ptr_bone", torch.zeros(1, dtype=torch.long))
    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_motion(self):
        for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_bone(self):
        for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_motion)
        self.queue_motion[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_motion[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_bone(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_bone)
        self.queue_bone[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_bone[0] = ptr

    def prior_gaussian(self, T):
        cluster = np.random.normal(59, 2, 1)
        cluster = int(np.round(cluster))
        if cluster > self.batch_size:
            cluster = self.batch_size
        cluster_index = np.random.choice(np.arange(self.nmb_prototypes), size=cluster, replace=False)  # [59]index
        add_index = np.random.choice(cluster_index, size=self.batch_size - cluster)
        cluster_index = np.concatenate((cluster_index, add_index), axis=0)  #
        np.random.shuffle(cluster_index)
        for i in range(self.batch_size):
            ax = np.arange(-cluster_index[i], self.nmb_prototypes - cluster_index[i])
            # ax = np.linalg.norm(ax, axis=0)
            T[:][i] = ax
        T = torch.tensor(T,dtype=torch.float32)
        T = F.normalize(T, dim=1)
        T = (1 / (self.sigm * np.sqrt(2 * np.pi))) * torch.exp(-T ** 2 / (2 * self.sigm * self.sigm))

        return T
    @torch.no_grad()
    def distributed_sinkhorn(self, out, sk_epsilon, sinkhorn_iterations):
        a, b = out.shape    #(64, 128)
        T = self.prior_gaussian(np.zeros([a, b]))
        T = T.to(out.device)

        Q = torch.exp(-out / (sk_epsilon + self.sk_T)) * torch.pow(T, self.sk_T / (sk_epsilon + self.sk_T))
        Q = Q.t()  #(128, 64)
        B = Q.shape[1]
        K = Q.shape[0]
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q /= torch.sum(Q,dim=0, keepdim=True)
        return Q.t()

    @torch.no_grad()
    def distributed_sinkhorn1(self, out, frame_embed, sk_epsilon, sinkhorn_iterations):
        # out (bs,K,N/F); frame_embed (bs,k,N/F)
        Q = torch.exp(-out/(sk_epsilon+self.sk_T)) * (torch.pow(frame_embed, self.sk_T/(sk_epsilon+self.sk_T)) + torch.tensor(1e-6))
        Q = Q.permute(0,2,1).contiguous() #(bs,128,64)
        B = Q.shape[2]  #
        K = Q.shape[1]  #

        sum_Q = torch.sum(Q, dim=[1, 2])
        Q /= sum_Q[:, None, None]
        for it in range(sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q /= torch.sum(Q, dim=1, keepdim=True)
        return Q.permute(0,2,1).contiguous()  #(bs, 128, 64)

    def forward(self, im_q, im_k=None, is_second_train=False, frameSK_or_SK=False, view='joint', knn_eval=False):
        if is_second_train:
            return self.second_train(im_q, im_k, frameSK_or_SK)

        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

        # Permute and Reshape
        N, C, T, V, M = im_q.size()
        im_q = im_q.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
        im_q_motion = im_q_motion.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
        im_q_bone = im_q_bone.permute(0, 2, 3, 1, 4).reshape(N, T, -1)

        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q, frameSK_or_SK, knn_eval)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion, frameSK_or_SK, knn_eval)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone, frameSK_or_SK, knn_eval)
            elif view == 'all':

                return (self.encoder_q(im_q, frameSK_or_SK, knn_eval) + \
                        self.encoder_q_motion(im_q_motion, frameSK_or_SK, knn_eval) + \
                        self.encoder_q_bone(im_q_bone, frameSK_or_SK, knn_eval)) / 3.
            else:
                raise ValueError

        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # Permute and Reshape
        im_k = im_k.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
        im_k_motion = im_k_motion.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
        im_k_bone = im_k_bone.permute(0, 2, 3, 1, 4).reshape(N, T, -1)

        # compute query features
        q = self.encoder_q(im_q, frameSK_or_SK)  # queries: NxC
        q = F.normalize(q, dim=1)

        q_motion = self.encoder_q_motion(im_q_motion, frameSK_or_SK)
        q_motion = F.normalize(q_motion, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone, frameSK_or_SK)
        q_bone = F.normalize(q_bone, dim=1)

        # compute key features for  s1 and  s2  skeleton representations
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()

            k = self.encoder_k(im_k, frameSK_or_SK)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion, frameSK_or_SK)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone, frameSK_or_SK)
            k_bone = F.normalize(k_bone, dim=1)

        # MOCO
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        # CMD loss
        lk_neg = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        lk_neg_motion = torch.einsum('nc,ck->nk', [k_motion, self.queue_motion.clone().detach()])
        lk_neg_bone = torch.einsum('nc,ck->nk', [k_bone, self.queue_bone.clone().detach()])

        # Top-k
        lk_neg_topk, topk_idx = torch.topk(lk_neg, self.topk, dim=-1)
        lk_neg_motion_topk, motion_topk_idx = torch.topk(lk_neg_motion, self.topk, dim=-1)
        lk_neg_bone_topk, bone_topk_idx = torch.topk(lk_neg_bone, self.topk, dim=-1)

        loss_cmd = loss_kld(torch.gather(l_neg_motion, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg_bone, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg, -1, motion_topk_idx) / self.student_T,
                            lk_neg_motion_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg_bone, -1, motion_topk_idx) / self.student_T,
                            lk_neg_motion_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg, -1, bone_topk_idx) / self.student_T,
                            lk_neg_bone_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg_motion, -1, bone_topk_idx) / self.student_T,
                            lk_neg_bone_topk / self.teacher_T)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits, logits_motion, logits_bone, labels, loss_cmd
    def second_train(self, im_q, im_k, frameSK_or_SK):
        if frameSK_or_SK:
            with torch.no_grad():
                w = self.prototype_q.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototype_q.weight.copy_(w)

                w_motion = self.prototype_q_motion.weight.data.clone()
                w_motion = nn.functional.normalize(w_motion, dim=1, p=2)
                self.prototype_q_motion.weight.copy_(w_motion)

                w_bone = self.prototype_q_bone.weight.data.clone()
                w_bone = nn.functional.normalize(w_bone, dim=1, p=2)
                self.prototype_q_bone.weight.copy_(w_bone)


            im_q_motion = torch.zeros_like(im_q)
            im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

            im_q_bone = torch.zeros_like(im_q)
            for v1, v2 in self.Bone:
                im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

            # Permute and Reshape
            N, C, T, V, M = im_q.size()
            im_q = im_q.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
            im_q_motion = im_q_motion.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
            im_q_bone = im_q_bone.permute(0, 2, 3, 1, 4).reshape(N, T, -1)

            im_k_motion = torch.zeros_like(im_k)
            im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

            im_k_bone = torch.zeros_like(im_k)
            for v1, v2 in self.Bone:
                im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

            # Permute and Reshape
            im_k = im_k.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
            im_k_motion = im_k_motion.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
            im_k_bone = im_k_bone.permute(0, 2, 3, 1, 4).reshape(N, T, -1)

            # compute query features
            q = self.encoder_q(im_q, frameSK_or_SK)  # queries: NxC
            q = F.normalize(q, dim=1)

            q_motion  = self.encoder_q_motion(im_q_motion, frameSK_or_SK)
            q_motion = F.normalize(q_motion, dim=1)

            q_bone = self.encoder_q_bone(im_q_bone, frameSK_or_SK)
            q_bone = F.normalize(q_bone, dim=1)


            qc = self.prototype_q(q)
            qc_motion = self.prototype_q_motion(q_motion)
            qc_bone = self.prototype_q_bone(q_bone)

            # compute key features for  s1 and  s2  skeleton representations
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                self._momentum_update_key_encoder_motion()
                self._momentum_update_key_encoder_bone()

                k = self.encoder_k(im_k, frameSK_or_SK)  # keys: NxC
                k = F.normalize(k, dim=1)

                k_motion = self.encoder_k_motion(im_k_motion, frameSK_or_SK)
                k_motion = F.normalize(k_motion, dim=1)

                k_bone = self.encoder_k_bone(im_k_bone, frameSK_or_SK)
                k_bone = F.normalize(k_bone, dim=1)

            kc = self.prototype_q(k)
            kc_motion = self.prototype_q_motion(k_motion)
            kc_bone = self.prototype_q_bone(k_bone)

            # MOCO
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
            l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

            l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
            l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)

            # apply temperature
            logits /= self.T
            logits_motion /= self.T
            logits_bone /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)
            self._dequeue_and_enqueue_motion(k_motion)
            self._dequeue_and_enqueue_bone(k_bone)

            ###############################################
            output_q = torch.cat([qc, qc_motion, qc_bone], dim=0)
            output_k = torch.cat([kc, kc_motion, kc_bone], dim=0)

            # output_k_sequece_embed = output_k_sequece_embed.mean(dim=-1)  #(3bs, 128)

            bs = len(im_q)
            sk_loss = 0
            for i, stream_id in enumerate([0, 1, 2]):
                with torch.no_grad():
                    # spatial
                    out = output_q[bs * stream_id: bs * (stream_id + 1)].detach()
                    q = self.distributed_sinkhorn(out, self.sk_epsilon, 3)  # args.sinkhorn_iterations

                subloss = 0
                for v in np.delete(np.arange(np.sum([1, 2])), stream_id):
                    x = output_k[bs * v: bs * (v + 1)] / self.sk_T
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                # loss += subloss / (np.sum([1,2]) - 1)
                sk_loss += subloss

            return logits, logits_motion, logits_bone, labels, sk_loss  # ,kl_loss
        else:
            with torch.no_grad():

                w_sequece = self.prototype_q_sequece.weight.data.clone()
                w_sequece = nn.functional.normalize(w_sequece, dim=1, p=2)
                self.prototype_q_sequece.weight.copy_(w_sequece)

                w_motion_sequece = self.prototype_q_motion_sequece.weight.data.clone()
                w_motion_sequece = nn.functional.normalize(w_motion_sequece, dim=1, p=2)
                self.prototype_q_motion_sequece.weight.copy_(w_motion_sequece)

                w_bone_sequece = self.prototype_q_bone_sequece.weight.data.clone()
                w_bone_sequece = nn.functional.normalize(w_bone_sequece, dim=1, p=2)
                self.prototype_q_bone_sequece.weight.copy_(w_bone_sequece)

            im_q_motion = torch.zeros_like(im_q)
            im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

            im_q_bone = torch.zeros_like(im_q)
            for v1, v2 in self.Bone:
                im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

            # Permute and Reshape
            N, C, T, V, M = im_q.size()
            im_q = im_q.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
            im_q_motion = im_q_motion.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
            im_q_bone = im_q_bone.permute(0, 2, 3, 1, 4).reshape(N, T, -1)

            im_k_motion = torch.zeros_like(im_k)
            im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

            im_k_bone = torch.zeros_like(im_k)
            for v1, v2 in self.Bone:
                im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

            # Permute and Reshape
            im_k = im_k.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
            im_k_motion = im_k_motion.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
            im_k_bone = im_k_bone.permute(0, 2, 3, 1, 4).reshape(N, T, -1)

            # compute query features
            q, q_sequece_embed = self.encoder_q(im_q, frameSK_or_SK)  # queries: NxC
            q = F.normalize(q, dim=1)
            q_sequece_embed = F.normalize(q_sequece_embed, dim=-1)

            q_motion, q_motion_sequece_embed = self.encoder_q_motion(im_q_motion, frameSK_or_SK)
            q_motion = F.normalize(q_motion, dim=1)
            q_motion_sequece_embed = F.normalize(q_motion_sequece_embed, dim=-1)

            q_bone, q_bone_sequece_embed = self.encoder_q_bone(im_q_bone, frameSK_or_SK)
            q_bone = F.normalize(q_bone, dim=1)
            q_bone_sequece_embed = F.normalize(q_bone_sequece_embed, dim=-1)


            qc_sequece = self.prototype_q_sequece(q_sequece_embed)
            qc_motion_sequece = self.prototype_q_motion_sequece(q_motion_sequece_embed)
            qc_bone_sequece = self.prototype_q_bone_sequece(q_bone_sequece_embed)  # (64,64,128)
            #Frame's index embed
            frame_embed = self.tem_embed(self.tem).squeeze(2)  # (bs,128,25->1,64)
            frame_embed = frame_embed.permute(0, 2, 1).contiguous()  # (bs,64,128)

            # compute key features for  s1 and  s2  skeleton representations
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                self._momentum_update_key_encoder_motion()
                self._momentum_update_key_encoder_bone()

                k, k_sequece_embed = self.encoder_k(im_k, frameSK_or_SK)  # keys: NxC
                k = F.normalize(k, dim=1)
                k_sequece_embed = F.normalize(k_sequece_embed, dim=1)

                k_motion, k_motion_sequece_embed = self.encoder_k_motion(im_k_motion, frameSK_or_SK)
                k_motion = F.normalize(k_motion, dim=1)
                k_motion_sequece_embed = F.normalize(k_motion_sequece_embed, dim=-1)

                k_bone, k_bone_sequece_embed = self.encoder_k_bone(im_k_bone, frameSK_or_SK)
                k_bone = F.normalize(k_bone, dim=1)
                k_bone_sequece_embed = F.normalize(k_bone_sequece_embed, dim=-1)

            kc_sequece = self.prototype_q_sequece(k_sequece_embed)
            kc_motion_sequece = self.prototype_q_motion_sequece(k_motion_sequece_embed)
            kc_bone_sequece = self.prototype_q_bone_sequece(k_bone_sequece_embed)

            # MOCO
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
            l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

            l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
            l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)

            # apply temperature
            logits /= self.T
            logits_motion /= self.T
            logits_bone /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)
            self._dequeue_and_enqueue_motion(k_motion)
            self._dequeue_and_enqueue_bone(k_bone)

            ###############################################

            output_q_sequece_embed = torch.cat([qc_sequece, qc_motion_sequece, qc_bone_sequece], dim=0)
            output_k_sequece_embed = torch.cat([kc_sequece, kc_motion_sequece, kc_bone_sequece], dim=0)  # (bs*3,64,128)

            bs = len(im_q)
            sk_loss1 = 0
            for i, stream_id in enumerate([0, 1, 2]):
                with torch.no_grad():

                    # temporal
                    out1 = output_q_sequece_embed[bs * stream_id: bs * (stream_id + 1)].detach()
                    q1 = self.distributed_sinkhorn1(out1, frame_embed, self.sk_epsilon, 3)  # (bs,64,128)
                    # q1 = F.avg_pool1d(q1, q1.size()[-1]).squeeze(-1)    #(bs/64, 128)

                subloss1 = 0
                for v in np.delete(np.arange(np.sum([1, 2])), stream_id):

                    x1 = output_k_sequece_embed[bs * v: bs * (v + 1)] / self.sk_T
                    loss_tmp = 0
                    for m in range(bs):
                        loss_tmp -= torch.mean(torch.sum(q1[m] * F.log_softmax(x1[m], dim=1), dim=1))
                    subloss1 += loss_tmp / bs
                # loss += subloss / (np.sum([1,2]) - 1)
                sk_loss1 += subloss1

            return logits, logits_motion, logits_bone, labels, sk_loss1  # ,kl_loss


