# metrics/saliency_metrics.py
import numpy as np
# from scipy.stats import entropy
# from sklearn.metrics import roc_auc_score

def normalize_map(map_):
    return (map_ - map_.min()) / (map_.max() - map_.min() + 1e-8)

# def auc(saliency_map, fixation_map):
#     return roc_auc_score(fixation_map.flatten(), saliency_map.flatten())

def nss(saliency_map, fixation_map):
    sm = (saliency_map - saliency_map.mean()) / saliency_map.std()
    return sm[fixation_map > 0].mean()

def cc(saliency_map, gt_map):
    sm = (saliency_map - saliency_map.mean()) / saliency_map.std()
    gt = (gt_map - gt_map.mean()) / gt_map.std()
    return np.corrcoef(sm.flatten(), gt.flatten())[0, 1]

# def kl_divergence(p_map, q_map):
    # p = p_map.flatten() / p_map.sum()
    # q = q_map.flatten() / q_map.sum()
    # return entropy(p, q)

def similarity(p_map, q_map):
    p = normalize_map(p_map)
    q = normalize_map(q_map)
    return np.sum(np.minimum(p, q))
