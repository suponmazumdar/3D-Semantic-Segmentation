import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.cluster import KMeans
import MinkowskiEngine as ME


def get_sp_feature(args, loader, model, current_growsp):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    model.eval()
    context = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(features, coords, device=0)

            # Model now returns (out_sparse, feats_F)
            # We need the normalized features (feats_F)
            _, feats = model(in_field)
            
            # feats is already normalized, shape [N, feats_dim]
            feats = feats[inds.long()]

            valid_mask = region!=-1
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:] * args.voxel_size
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            if current_growsp is not None:
                region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
                region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
                region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

                rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats = F.normalize(region_feats, dim=-1)
                region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
                #
                if region_feats.size(0)<current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]
            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            '''Compute avg rgb/pfh for each Superpoints to help Primitives Learning'''
            final_rgb = F.linear(neural_region_corr.t(), pc_rgb.t())/per_neural_region_num
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask].cpu()).unsqueeze(0).cuda())

            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            feats = torch.cat((feats, args.c_rgb*final_rgb, args.c_shape*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sp_index.append(neural_region)
            context.append((scene_name, gt, raw_region))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
    return point_feats_list, point_labels_list, all_sp_index, context



def get_kittisp_feature(args, loader, model, current_growsp):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    model.eval()
    context = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(coords[:, 1:]*args.voxel_size, coords, device=0)

            # Model now returns (out_sparse, feats_F)
            _, feats = model(in_field)
            
            feats = feats[inds.long()]

            valid_mask = region!=-1
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_remission = features
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            if current_growsp is not None:
                region_feats = F.normalize(region_feats, dim=-1)
                #
                if region_feats.size(0) < current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]
            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            final_remission = F.linear(neural_region_corr.t(), pc_remission.t())/per_neural_region_num
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)
            #
            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask]).unsqueeze(0))

            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            feats = torch.cat((feats, args.c_rgb*final_remission, args.c_shape*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sp_index.append(neural_region)
            context.append((scene_name, gt, raw_region))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return point_feats_list, point_labels_list, all_sp_index, context



def get_pseudo(args, context, cluster_pred, all_sp_index=None):
    print('computing pseduo labels...')
    pseudo_label_folder = args.pseudo_label_path + '/'
    if not os.path.exists(pseudo_label_folder):
        os.makedirs(pseudo_label_folder)
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in range(len(context)):
        scene_name, labels, region = context[i]

        sub_cluster_pred = all_sp_index[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = labels[valid_mask]
        pseudo_gt = -torch.ones_like(labels)
        pseudo_gt_tmp = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(labels.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                pseudo_gt_tmp[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = pseudo_gt_tmp
        #
        pc_no += 1
        new_region = np.unique(sub_cluster_pred)
        region_num += len(new_region[new_region != -1])

        pseudo_label_file = pseudo_label_folder + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    return all_pseudo, all_gt, all_pseudo_gt


def get_pseudo_kitti(args, context, cluster_pred, all_sub_cluster=None):
    print('computing pseduo labels...')
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in range(len(context)):
        scene_name, labels, region = context[i]

        sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = labels[valid_mask]
        pseudo_gt = -torch.ones_like(labels)
        pseudo_gt_tmp = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(labels.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                pseudo_gt_tmp[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = pseudo_gt_tmp
        #
        pc_no += 1
        new_region = np.unique(sub_cluster_pred)
        region_num += len(new_region[new_region != -1])

        pseudo_label_folder = args.pseudo_label_path + '/' + scene_name[0:3]
        if not os.path.exists(pseudo_label_folder):
            os.makedirs(pseudo_label_folder)

        pseudo_label_file = args.pseudo_label_path + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    return all_pseudo, all_gt, all_pseudo_gt


def get_fixclassifier(in_channel, centroids_num, centroids):
    classifier = nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
    centroids = F.normalize(centroids, dim=1)
    classifier.weight.data = centroids
    for para in classifier.parameters():
        para.requires_grad = False
    return classifier


def compute_hist(normal, bins=10, min=-1, max=1):
    ## normal : [N, 3]
    normal = F.normalize(normal)
    relation = torch.mm(normal, normal.t())
    relation = torch.triu(relation, diagonal=0) # top-half matrix
    hist = torch.histc(relation, bins, min, max)
    # hist = torch.histogram(relation, bins, range=(-1, 1))
    hist /= hist.sum()

    return hist


def load_model_from_state_dict(model, state_dict, strip_module_prefix=True):
    """
    Load a state_dict into model robustly:
    - strips 'module.' prefix if present
    - only copies keys that exist in model and match shape
    - uses strict=False so missing keys are ignored
    Returns the model (moved to same device as before).
    """
    import re
    model_state = model.state_dict()
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if strip_module_prefix and k.startswith("module."):
            nk = k[len("module."):]
        if nk in model_state and model_state[nk].shape == v.shape:
            new_state[nk] = v
    # load what matches
    model.load_state_dict(new_state, strict=False)
    return model


def update_teacher_weights(model_t, model_s, alpha=0.999, update_bn=True):
    """
    EMA update for teacher <- alpha * teacher + (1-alpha) * student.
    Copies BN buffers exactly if update_bn=True.
    Leaves teacher in eval mode and freezes its params.
    """
    model_t.train()
    with torch.no_grad():
        s_params = dict(model_s.named_parameters())
        for n, p_t in model_t.named_parameters():
            p_s = s_params.get(n, None)
            if p_s is None:
                continue
            p_t.data.mul_(alpha).add_(p_s.data * (1.0 - alpha))
        if update_bn:
            # copy bn buffers (running_mean/var)
            s_buffers = dict(model_s.named_buffers())
            for n, b_t in model_t.named_buffers():
                b_s = s_buffers.get(n, None)
                if b_s is None:
                    continue
                try:
                    b_t.data.copy_(b_s.data)
                except Exception:
                    pass
    for p in model_t.parameters():
        p.requires_grad = False
    model_t.eval()
    return model_t


def get_aug_data(coords, features, normals, voxel_size=0.05, aug_strength='weak', device=None):
    """
    Augmentation helper used by the TS pipeline.
    Input:
      - coords: Tensor[N,4] (batch_idx, x, y, z) where x,y,z are continuous coords
      - features: Tensor[N, C] (features[:,3:6] correspond to xyz/voxel conventions in repo)
      - normals: Tensor[N, 3]
    Returns:
      - coords_voxel (Tensor[N,4] int32 voxel coordinates),
      - features_aug (Tensor[N, C]),
      - normals_aug (Tensor[N, 3])
    """
    device = coords.device if device is None else device
    coords = coords.to(device)
    features = features.to(device).float()
    normals = normals.to(device).float()

    batch_idx = coords[:, 0].long().unsqueeze(1)  # [N,1]
    coords_xyz = coords[:, 1:].float()  # continuous XYZ

    if aug_strength == 'weak':
        angle = float((torch.rand(1, device=device).item() - 0.5) * (2 * torch.pi / 64))
        jitter = 0.001
        scale = 1.0 + float(torch.randn(1, device=device).item() * 0.001)
    else:
        angle = float((torch.rand(1, device=device).item() - 0.5) * (2 * torch.pi / 4))
        jitter = 0.005
        scale = float(1.0 + (torch.rand(1, device=device).item() - 0.5) * 0.2)

    cosv = float(torch.cos(torch.tensor(angle, device=device)))
    sinv = float(torch.sin(torch.tensor(angle, device=device)))
    rot = torch.tensor([[cosv, -sinv, 0.0], [sinv, cosv, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float)

    coords_aug = torch.matmul(coords_xyz, rot.t()) * scale
    coords_aug = coords_aug + torch.randn_like(coords_aug) * jitter

    normals_aug = torch.matmul(normals, rot.t())
    normals_aug = F.normalize(normals_aug, dim=1)

    # update features' xyz part: convention in repo is features[:,3:] * voxel_size == real XYZ
    features_aug = features.clone()
    if features_aug.size(1) >= 6:
        features_aug[:, 3:6] = coords_aug / float(voxel_size)

    # voxelize coords consistently (floor) and ensure int32 dtype
    vox = torch.floor(coords_aug / float(voxel_size)).int()  # Changed to .int() for int32
    coords_voxel = torch.cat((batch_idx.int(), vox), dim=1).to(device)  # Ensure batch_idx is also int32

    return coords_voxel, features_aug, normals_aug

def compute_nt_xent_loss(z_i, z_j, temperature=0.1, eps=1e-8):
    """
    NT-Xent (InfoNCE) loss between two batches of embeddings z_i and z_j.
    z_i, z_j: [N, D] (raw or already normalized). Returns scalar.
    """
    assert z_i.shape == z_j.shape
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
    sim = torch.matmul(z, z.t()) / temperature  # [2N, 2N]
    # mask out self-similarities
    diag = torch.eye(2 * N, device=sim.device).bool()
    sim.masked_fill_(diag, -9e15)
    # positives: i with i+N
    pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)  # [2N]
    denom = torch.logsumexp(sim, dim=1)  # [2N]
    loss = -pos + denom
    return loss.mean()
