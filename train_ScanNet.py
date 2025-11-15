import argparse
import time
import os
import numpy as np
import random
from datasets.ScanNet import Scannettrain, cfl_collate_fn
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.fpn import Res16FPN18, PredictionHead # Import PredictionHead
from eval_ScanNet import eval
from lib.utils import get_pseudo, get_sp_feature, get_fixclassifier, update_teacher_weights, get_aug_data, load_model_from_state_dict, compute_nt_xent_loss
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, MeanShift
import logging
from os.path import join
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.optim.lr_scheduler import LambdaLR
warnings.filterwarnings('ignore')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/ScanNet/processed/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= 'data/ScanNet/initial_superpoints/',
                        help='initial sp path')
    ###
    parser.add_argument('--save_path', type=str, default='ckpt/ScanNet/',
                        help='model savepath')
    parser.add_argument('--max_epoch', type=list, default=[70,70], help='max epoch')
    parser.add_argument('--max_iter', type=list, default=[50000, 100000], help='max iter')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD parameters')
    parser.add_argument('--dampening', type=float, default=0.1, help='SGD parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='SGD parameters')
    parser.add_argument('--workers', type=int, default=8, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--log-interval', type=int, default=150, help='log interval')
    parser.add_argument('--batch_size', type=int, default=2, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=20, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--pseudo_label_path', default='pseudo_label_scannet/', type=str, help='pseudo label save path')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--growsp_start', type=int, default=80, help='the start number of growing superpoint')
    parser.add_argument('--growsp_end', type=int, default=30, help='the end number of grwoing superpoint')
    parser.add_argument('--drop_threshold', type=int, default=30, help='ignore superpoints with few points')
    parser.add_argument('--w_rgb', type=float, default=5/5, help='weight for RGB in merging superpoint')
    parser.add_argument('--w_xyz', type=float, default=1/5, help='weight for XYZ in merging superpoint')
    parser.add_argument('--w_norm', type=float, default=4/5, help='weight for Normal in merging superpoint')
    parser.add_argument('--c_rgb', type=float, default=3, help='weight for RGB in clustering primitives')
    parser.add_argument('--c_shape', type=float, default=3, help='weight for PFH in clustering primitives')
    parser.add_argument('--ts_lambda', type=float, default=0.1, help='weight for contrastive loss')
    parser.add_argument('--ts_momentum', type=float, default=0.999, help='EMA momentum for teacher update')
    return parser.parse_args()



def main(args, logger):

    '''Prepare Data'''
    trainset = Scannettrain(args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=cfl_collate_fn(), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))
    clusterset = Scannettrain(args)
    cluster_loader = DataLoader(clusterset, batch_size=1, collate_fn=cfl_collate_fn(), num_workers=args.cluster_workers, pin_memory=True)

    '''Prepare Model/Optimizer'''
    # Student Network (S)
    model_s = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args)
    logger.info("Student Model:")
    logger.info(model_s)
    model_s = model_s.cuda()
    
    # teacher Network (T)
    model_t = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args)
    model_t = load_model_from_state_dict(model_t, model_s.state_dict()).cuda()
    # Freeze teacher weights
    for param in model_t.parameters():
        param.requires_grad = False
        
    # prediction Head for Student 
    pred_head = PredictionHead(in_channels=args.feats_dim, out_channels=args.feats_dim, D=3).cuda()
    # Teacher Projector
    teacher_proj_head = PredictionHead(in_channels=args.feats_dim, out_channels=args.feats_dim, D=3).cuda()
    teacher_proj_head = load_model_from_state_dict(teacher_proj_head, pred_head.state_dict()).cuda()
    for param in teacher_proj_head.parameters():
        param.requires_grad = False
    
    # optimizer for student and prediction head
    optimizer = torch.optim.SGD(list(model_s.parameters()) + list(pred_head.parameters()), 
                            lr=args.lr, momentum=args.momentum, dampening=args.dampening, 
                            weight_decay=args.weight_decay)
    scheduler = PolyLR(optimizer, max_iter=args.max_iter[0])
    
    # Loss functions
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda() # Clustering loss
    loss_mse = torch.nn.MSELoss().cuda() # Contrastive loss
    
    start_grow_epoch = 0
    
    '''Train and Cluster'''
    is_Growing = False
    a = time.time()
    for epoch in range(1, args.max_epoch[0] + 1):
        '''Take 10 epochs as a round'''
        if (epoch - 1) % 10 == 0:
            classifier = cluster(args, logger, cluster_loader, model_s, epoch, start_grow_epoch, is_Growing)
        
        train(train_loader, logger, model_s, model_t, pred_head, optimizer, loss_ce, loss_mse, 
      epoch, scheduler, classifier, args.ts_momentum, args.ts_lambda, args)


        if epoch % 10== 0:
            torch.save(model_s.state_dict(), join(args.save_path,  'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(classifier.state_dict(), join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth'))
            
            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch, args) 
                logger.info('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)

            iterations = (epoch) * len(train_loader) 
            if iterations > args.max_iter[0]:
                start_grow_epoch = epoch
                break

    '''Superpoints will grow in 2nd Stage'''
    logger.info('#################################')
    logger.info('### Superpoints Begin Grwoing ###')
    logger.info('#################################')
    is_Growing = True
    # Re-initialize optimizer for the second stage
    optimizer = torch.optim.SGD(
        list(model_s.parameters()) + list(pred_head.parameters()), 
        lr=args.lr, momentum=args.momentum, dampening=args.dampening, 
        weight_decay=args.weight_decay
    )
    scheduler = PolyLR(optimizer, max_iter=args.max_iter[1])
    for epoch in range(1, args.max_epoch[1] + 1):
        epoch += start_grow_epoch

        if (epoch - 1) % 10 == 0:
            classifier = cluster(args, logger, cluster_loader, model_s, epoch, start_grow_epoch, is_Growing)

        train(train_loader, logger, model_s, model_t, pred_head, optimizer, loss_ce, loss_mse, 
              epoch, scheduler, classifier, args.ts_momentum, args.ts_lambda, args)

        if epoch % 10 == 0:
            torch.save(model_s.state_dict(), join(args.save_path,  'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(classifier.state_dict(), join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth'))
            
            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch, args)
                logger.info('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)


def hungarian_matched_mIoU(gt_labels, pred_clusters, num_classes):
    """Compute mIoU after optimal cluster→GT remapping."""
    mask = (gt_labels != -1)
    gt = gt_labels[mask]
    pred = pred_clusters[mask]

    K = pred.max() + 1
    conf_mat = np.zeros((num_classes, K), dtype=np.int64)
    for i in range(len(gt)):
        conf_mat[gt[i], pred[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    pred_aligned = np.array([mapping.get(p, -1) for p in pred])

    IoUs = []
    for c in range(num_classes):
        gt_mask = (gt == c)
        pred_mask = (pred_aligned == c)
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union > 0:
            IoUs.append(inter / union)
    mIoU = np.mean(IoUs) * 100
    return mIoU, IoUs, mapping


def cluster(args, logger, cluster_loader, model, epoch, start_grow_epoch=None, is_Growing=False):
    """Perform clustering and compute pseudo labels."""
    time_start = time.time()
    cluster_loader.dataset.mode = 'cluster'

    current_growsp = None
    if is_Growing:
        current_growsp = int(
            args.growsp_start - ((epoch - start_grow_epoch) / args.max_epoch[1]) *
            (args.growsp_start - args.growsp_end)
        )
        current_growsp = max(current_growsp, args.growsp_end)
        logger.info('Epoch: {}, Superpoints Grow to {}'.format(epoch, current_growsp))

    # extractting  features per superpoint
    feats, labels, sp_index, context = get_sp_feature(args, cluster_loader, model, current_growsp)
    sp_feats = torch.cat(feats, dim=0)

    # K-Means clustering on superpoint features
    primitive_labels = KMeans(
        n_clusters=args.primitive_num, n_jobs=-1
    ).fit_predict(sp_feats.numpy().astype(np.float32))
    sp_feats = sp_feats[:, :args.feats_dim]  

    # Compute primitive centers
    primitive_centers = torch.zeros((args.primitive_num, args.feats_dim))
    for cid in range(args.primitive_num):
        mask_c = primitive_labels == cid
        if mask_c.sum() > 0:
            primitive_centers[cid] = sp_feats[mask_c].mean(0, keepdims=True)
    primitive_centers = F.normalize(primitive_centers, dim=1)
    classifier = get_fixclassifier(
        in_channel=args.feats_dim, centroids_num=args.primitive_num, centroids=primitive_centers
    )

    # Generate pseudo labels
    all_pseudo, all_gt, all_pseudo_gt = get_pseudo(args, context, primitive_labels, sp_index)
    logger.info('labelled points ratio %.2f clustering time: %.2fs',
                (all_pseudo != -1).sum() / all_pseudo.shape[0],
                time.time() - time_start)

    # Superpoint-level IoU
    sem_num = args.semantic_class
    mask = (all_pseudo_gt != -1)
    histogram = np.bincount(
        sem_num * all_gt.astype(np.int32)[mask] + all_pseudo_gt.astype(np.int32)[mask],
        minlength=sem_num ** 2
    ).reshape(sem_num, sem_num)

    o_Acc = histogram[np.arange(sem_num), np.arange(sem_num)].sum() / histogram.sum() * 100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('Superpoints oAcc {:.2f} IoUs'.format(o_Acc) + s)

    # Primitive-level IoU (before Hungarian)
    pseudo_class2gt = -np.ones_like(all_gt)
    for i in range(args.primitive_num):
        mask_c = all_pseudo == i
        if mask_c.sum() > 0:
            pseudo_class2gt[mask_c] = torch.mode(torch.from_numpy(all_gt[mask_c])).values
    mask = (pseudo_class2gt != -1) & (all_gt != -1)
    histogram = np.bincount(
        sem_num * all_gt.astype(np.int32)[mask] + pseudo_class2gt.astype(np.int32)[mask],
        minlength=sem_num ** 2
    ).reshape(sem_num, sem_num)

    o_Acc = histogram[np.arange(sem_num), np.arange(sem_num)].sum() / histogram.sum() * 100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('Primitives oAcc {:.2f} IoUs (raw cluster alignment)'.format(o_Acc) + s)

    # Hungarian-matched mIoU 
    hung_mIoU, hung_IoUs, mapping = hungarian_matched_mIoU(
        all_gt.astype(np.int32), all_pseudo_gt.astype(np.int32), sem_num
    )
    logger.info('Hungarian-matched mIoU: {:.2f}%'.format(hung_mIoU))
    logger.info('Cluster→GT mapping (partial): {}'.format(
        dict(list(mapping.items())[:10])
    ))

    return classifier.cuda()


def train(train_loader, logger, model_s, model_t, pred_head, optimizer, loss_ce, loss_mse, 
          epoch, scheduler, classifier, alpha_momentum, lambda_contrastive, args):
    """
    Teacher-Student training following BYOL-style framework:
    - Student processes strong augmentation and makes prediction
    - Teacher processes weak augmentation (or strong) as target
    - Predictor head only on student side
    - Loss: negative cosine similarity + clustering loss
    """
    train_loader.dataset.mode = 'train'
    model_s.train()
    pred_head.train()
    model_t.eval()
    
    loss_display = 0
    loss_sem_display = 0
    loss_contr_display = 0
    time_curr = time.time()
    
    for batch_idx, data in enumerate(train_loader):
        iteration = (epoch - 1) * len(train_loader) + batch_idx + 1

        coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data
        coords = coords.cuda()
        features = features.cuda()
        normals = normals.cuda()
        pseudo_labels = pseudo_labels.long().cuda()

        # 1. CLUSTERING LOSS 
        in_field = ME.TensorField(features.float(), coords.int(), device=0)
        out_sparse, feats_s = model_s(in_field)
        
        # Classification loss on normalized features
        inds_long = inds.long()
        inds_long = torch.clamp(inds_long, 0, len(pseudo_labels) - 1)
        selected_pseudo_labels = pseudo_labels[inds_long]
        feats_ce = feats_s[inds_long]
        logits = F.linear(feats_ce, F.normalize(classifier.weight))
        num_classes = logits.shape[1]

        valid_mask = (selected_pseudo_labels >= 0) & (selected_pseudo_labels < num_classes)
        if valid_mask.sum() > 0:
            loss_sem = loss_ce(logits[valid_mask] * 5, selected_pseudo_labels[valid_mask]).mean()
        else:
            loss_sem = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 2. TEACHER-STUDENT CONTRASTIVE LOSS
        # creating two augmented views
        coords_s, features_s_aug, normals_s = get_aug_data(
            coords.clone(), features.clone(), normals.clone(), 
            voxel_size=args.voxel_size, aug_strength='strong'
        )
        coords_t, features_t_aug, normals_t = get_aug_data(
            coords.clone(), features.clone(), normals.clone(), 
            voxel_size=args.voxel_size, aug_strength='weak'
        )
        
        # Student forward 
        in_field_s = ME.TensorField(features_s_aug.float(), coords_s.int(), device=0)
        out_sparse_s, feats_s_aug = model_s(in_field_s)
        
        # Prediction head on student
        pred_s = pred_head(out_sparse_s)
        pred_s_feats = F.normalize(pred_s.F, dim=1)
        
        # Teacher forward ,stop gradient
        with torch.no_grad():
            in_field_t = ME.TensorField(features_t_aug.float(), coords_t.int(), device=0)
            out_sparse_t, feats_t_aug = model_t(in_field_t)
            target_feats = F.normalize(feats_t_aug.detach(), dim=1)
        
        # Contrastive loss: negative cosine similarity
        # This encourages student prediction to match teacher features
        loss_contrastive = 2 - 2 * (pred_s_feats * target_feats).sum(dim=1).mean()

        # 3. TOTAL LOSS AND OPTIMIZATION
        loss_total = loss_sem + lambda_contrastive * loss_contrastive
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        
        # 4. EMA UPDATE FOR TEACHER
        update_teacher_weights(model_t, model_s, alpha=alpha_momentum)
        
        # 5. LOGGING
        loss_display += loss_total.item()
        loss_sem_display += loss_sem.item()
        loss_contr_display += loss_contrastive.item()

        if (batch_idx + 1) % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            loss_sem_display /= args.log_interval
            loss_contr_display /= args.log_interval
            
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Total: {:.4f}, Clust: {:.4f}, Contr: {:.4f}, lr: {:.3e}, Time: {:.2f}s'.format(
                    epoch, (batch_idx + 1), len(train_loader), 
                    100. * (batch_idx + 1) / len(train_loader),
                    iteration, loss_display, loss_sem_display, loss_contr_display, 
                    scheduler.get_lr()[0], time_used
                ))
            
            time_curr = time.time()
            loss_display = 0
            loss_sem_display = 0
            loss_contr_display = 0



def compute_cosine_similarity_loss(pred, target):
    """
    Compute negative cosine similarity loss (like BYOL/SimSiam).
    pred, target: [N, D] normalized tensors
    Returns: scalar loss
    """
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    loss = 2 - 2 * (pred * target).sum(dim=1).mean()
    return loss


from torch.optim.lr_scheduler import LambdaLR

class LambdaStepLR(LambdaLR):
  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v

class PolyLR(LambdaStepLR):
  def __init__(self, optimizer, max_iter=50000, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)

def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

def set_seed(seed):
   
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(torch.cuda.current_device()) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    args = parse_args()

    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = set_logger(os.path.join(args.save_path, 'train.log'))

    seed = args.seed
    set_seed(seed)

    main(args, logger)