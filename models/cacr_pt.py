import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps

from models.ssm_layer import FFCR
from models.vit.PointTransformerV3.PTV3 import PointTransformerV3
from models.foreground_align import FC_AMMD

class CACRFS3D(nn.Module):
    """
    Class Conditional Distribution Alignment and Collaborative Restructuring for Few-shot Point Cloud Semantic Segmentation.
    """
    def __init__(self, args):
        super(CACRFS3D, self).__init__()
        # self.gpu_id = args.gpu_id
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.n_subprototypes = args.n_subprototypes
        self.k_connect = args.k_connect
        self.sigma = args.sigma

        self.mix_dim = 64
        self.feat_dim = args.output_dim

        self.n_classes = self.n_way + 1

        self.max_per_class = args.max_per_class
        self.sigma_multipliers = args.sigma_multipliers
        self.num_neg = args.num_neg
        self.gamma_margin = args.gamma_margin
        self.lambda_mmd = args.lambda_ammd

        self.encoder = PointTransformerV3(in_channels=3, enc_depths=(2, 2, 2, 6, 2),
                                          enc_channels=(32, 64, 128, 256, 512), dec_depths=(2, 2, 2, 2),
                                          dec_channels=(64, 64, 128, 256))

        self.ffcr_learner = FFCR(cin=self.mix_dim, cout=self.feat_dim, d_model=args.m_dim, d_state=args.state_dim,
                                 headdim=args.head_dim, chunk_size=args.chunk_size)

        # Foreground Class-conditional Adaptive Multi-kernel Maximum Mean Discrepancy
        self.fc_ammd_loss = FC_AMMD(n_way=self.n_way, max_per_class=self.max_per_class,
                                    sigma_multipliers=self.sigma_multipliers, num_neg=self.num_neg,
                                    gamma=self.gamma_margin)

    def forward(self, support_x, support_y, query_x, query_y, Aux=True):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
            Aux: True for FC-AMMD Auxiliary in the training phase.
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way * self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat = self.getFeatures(query_x)

        support_feat = support_feat.transpose(1, 2).contiguous().view(self.n_way, -1, self.k_shot * self.n_points)

        # Foreground Class-conditional Adaptive Multi-kernel Maximum Mean Discrepancy
        if Aux:
            support_y_ = support_y.view(self.n_way, -1)
            loss_fc_ammd = self.fc_ammd_loss(support_feat, support_y_, query_feat, query_y, self.lambda_mmd)

        # Fine-grained Feature Collaborative Restructure for support & query features
        mix_feature = torch.cat((support_feat, query_feat), dim=-1)
        mix_feature = self.ffcr_learner(mix_feature)
        mix_support_feature = mix_feature[:, :, :self.k_shot * self.n_points]  # re-support
        mix_support_feature = mix_support_feature.view(self.n_way, -1, self.k_shot, self.n_points)
        mix_support_feature = mix_support_feature.transpose(1, 2).contiguous()
        mix_query_feature = mix_feature[:, :, self.k_shot * self.n_points:]  # re-query
        mix_query_feature = mix_query_feature.transpose(1, 2).contiguous().view(-1, self.feat_dim)

        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        fg_prototypes, fg_labels = self.getForegroundPrototypes(mix_support_feature, fg_mask, k=self.n_subprototypes)
        bg_prototype, bg_labels = self.getBackgroundPrototypes(mix_support_feature, bg_mask, k=self.n_subprototypes)

        # prototype learning
        if bg_prototype is not None and bg_labels is not None:
            prototypes = torch.cat((bg_prototype, fg_prototypes), dim=0)  # (*, feat_dim)
            prototype_labels = torch.cat((bg_labels, fg_labels), dim=0)  # (*,n_classes)
        else:
            prototypes = fg_prototypes
            prototype_labels = fg_labels
        self.num_prototypes = prototypes.shape[0]

        # construct label matrix Y, with Y_ij = 1 if x_i is from the support set and labeled as y_i = j, otherwise Y_ij = 0.
        self.num_nodes = self.num_prototypes + mix_query_feature.shape[0]  # number of node of partial observed graph
        Y = torch.zeros(self.num_nodes, self.n_classes).cuda()
        Y[:self.num_prototypes] = prototype_labels

        # construct feat matrix F
        node_feat = torch.cat((prototypes, mix_query_feature), dim=0)  # (num_nodes, feat_dim)

        # label propagation
        A = self.calculateLocalConstrainedAffinity(node_feat, k=self.k_connect)
        Z = self.label_propagate(A, Y)  # (num_nodes, n_way+1)

        query_pred = Z[self.num_prototypes:, :]  # (n_queries*num_points, n_way+1)
        query_pred = query_pred.view(-1, query_y.shape[1], self.n_classes).transpose(1,
                                                                                     2)  # (n_queries, n_way+1, num_points)
        loss = self.computeCrossEntropyLoss(query_pred, query_y)

        if Aux:
            loss += loss_fc_ammd

        return query_pred, loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        B, C, N = x.shape
        pc_feat = self.encoder(x)
        x_feat = pc_feat["feat"].view(B, N, -1).permute(0, 2, 1)

        return x_feat

    def getMutiplePrototypes(self, feat, k):
        """
        Extract multiple prototypes by points separation and assembly

        Args:
            feat: input point features, shape:(n_points, feat_dim)
        Return:
            prototypes: output prototypes, shape: (n_prototypes, feat_dim)
        """
        # sample k seeds as initial centers with Farthest Point Sampling (FPS)
        n = feat.shape[0]
        assert n > 0
        ratio = k / n
        if ratio < 1:
            fps_index = fps(feat, None, ratio=ratio, random_start=False).unique()
            num_prototypes = len(fps_index)
            farthest_seeds = feat[fps_index]

            # compute the point-to-seed distance
            distances = F.pairwise_distance(feat[:, None, :], farthest_seeds[None, :, :],
                                            p=2)  # (n_points, n_prototypes)

            # hard assignment for each point
            assignments = torch.argmin(distances, dim=1)  # (n_points,)

            # aggregating each cluster to form prototype
            prototypes = torch.zeros((num_prototypes, self.feat_dim)).cuda()
            for i in range(num_prototypes):
                selected = torch.nonzero(assignments == i).squeeze(1)
                selected = feat[selected, :]
                prototypes[i] = selected.mean(0)
            return prototypes
        else:
            return feat

    def getForegroundPrototypes(self, feats, masks, k=100):
        """
        Extract foreground prototypes for each class via clustering point features within that class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: foreground binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: foreground prototypes, shape: (n_way*k, feat_dim)
            labels: foreground prototype labels (one-hot), shape: (n_way*k, n_way+1)
        """
        prototypes = []
        labels = []
        for i in range(self.n_way):
            # extract point features belonging to current foreground class
            feat = feats[i, ...].transpose(1, 2).contiguous().view(-1, self.feat_dim)  # (k_shot*num_points, feat_dim)
            index = torch.nonzero(masks[i, ...].view(-1)).squeeze(1)  # (k_shot*num_points,)
            feat = feat[index]
            class_prototypes = self.getMutiplePrototypes(feat, k)
            prototypes.append(class_prototypes)

            # construct label matrix
            class_labels = torch.zeros(class_prototypes.shape[0], self.n_classes)
            class_labels[:, i + 1] = 1
            labels.append(class_labels)

        prototypes = torch.cat(prototypes, dim=0)
        labels = torch.cat(labels, dim=0)

        return prototypes, labels

    def getBackgroundPrototypes(self, feats, masks, k=100):
        """
        Extract background prototypes via clustering point features within background class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: background binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: background prototypes, shape: (k, feat_dim)
            labels: background prototype labels (one-hot), shape: (k, n_way+1)
        """
        feats = feats.transpose(2, 3).contiguous().view(-1, self.feat_dim)
        index = torch.nonzero(masks.view(-1)).squeeze(1)
        feat = feats[index]
        # in case this support set does not contain background points..
        if feat.shape[0] != 0:
            prototypes = self.getMutiplePrototypes(feat, k)

            labels = torch.zeros(prototypes.shape[0], self.n_classes)
            labels[:, 0] = 1

            return prototypes, labels
        else:
            return None, None

    def calculateLocalConstrainedAffinity(self, node_feat, k=200, method='gaussian'):
        """
        Calculate the Affinity matrix of the nearest neighbor graph constructed by prototypes and query points,
        It is a efficient way when the number of nodes in the graph is too large.

        Args:
            node_feat: input node features
                  shape: (num_nodes, feat_dim)
            k: the number of nearest neighbors for each node to compute the similarity
            method: 'cosine' or 'gaussian', different similarity function
        Return:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
        """
        # kNN search for the graph
        X = node_feat.detach().cpu().numpy()
        # build the index with cpu version
        index = faiss.IndexFlatL2(self.feat_dim)
        index.add(X)
        _, I = index.search(X, k + 1)
        I = torch.from_numpy(I[:, 1:]).cuda()  # (num_nodes, k)

        # create the affinity matrix
        knn_idx = I.unsqueeze(2).expand(-1, -1, self.feat_dim).contiguous().view(-1, self.feat_dim)
        knn_feat = torch.gather(node_feat, dim=0, index=knn_idx).contiguous().view(self.num_nodes, k, self.feat_dim)

        if method == 'cosine':
            knn_similarity = F.cosine_similarity(node_feat[:, None, :], knn_feat, dim=2).cuda()
        elif method == 'gaussian':
            dist = F.pairwise_distance(node_feat[:, None, :], knn_feat, p=2)
            knn_similarity = torch.exp(-0.5 * (dist / self.sigma) ** 2).cuda()
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)

        A = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float).cuda()
        A.scatter_(1, I, knn_similarity)
        A = A + A.transpose(0, 1)

        identity_matrix = torch.eye(self.num_nodes, requires_grad=False).cuda()
        A = A * (1 - identity_matrix)
        return A

    def label_propagate(self, A, Y, alpha=0.99):
        """ Label Propagation, refer to "Learning with Local and Global Consistency" NeurIPs 2003
        Args:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
            Y: initial label matrix, shape: (num_nodes, n_way+1)
            alpha: a parameter to control the amount of propagated info.
        Return:
            Z: label predictions, shape: (num_nodes, n_way+1)
        """
        # compute symmetrically normalized matrix S
        eps = np.finfo(float).eps
        D = A.sum(1)  # (num_nodes,)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D_sqrt_inv = torch.diag_embed(D_sqrt_inv).cuda()
        S = D_sqrt_inv @ A @ D_sqrt_inv

        # close form solution
        Z = torch.inverse(torch.eye(self.num_nodes).cuda() - alpha * S + eps) @ Y
        return Z

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)
