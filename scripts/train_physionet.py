import os
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import pickle
import glob
import tarfile
import io
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from addp import (
    ADDPCalibration,
    apply_addp_routing_calibrated,
    calibration_auxiliary_loss,
)
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============ 1. Args ============
class Args:
    def __init__(self, device):
        self.device = device
        self.hid_dim = 64
        self.ndim = 41
        self.history = 24.0
        self.patch_size = 8.0
        self.stride = 8.0
        self.npatch = int(np.ceil((self.history - self.patch_size) / self.stride)) + 1
        self.nlayer = 1
        self.te_dim = 10
        self.node_dim = 10
        self.nhead = 2
        self.tf_layer = 1
        self.hop = 2
        self.outlayer = "Linear"
        self.drop_ratio = 0.0


class nconv(nn.Module):
    def __init__(self): super(nconv,self).__init__()
    def forward(self, x, A): return torch.einsum('bfnm,bmnv->bfvm',(x,A)).contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)
    def forward(self, x): return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        self.mlp = linear((order*support_len+1)*c_in, c_out)
        self.order = order
    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a); out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a); out.append(x2); x1 = x2
        h = torch.cat(out, dim=1)
        return F.relu(self.mlp(h))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class ADDPModel(nn.Module):
    def __init__(self, args, supports=None, dropout=0, use_future_timestamps=True):
        super(ADDPModel, self).__init__()
        self.use_future_timestamps = use_future_timestamps
        self.N, self.M, self.n_layer = args.ndim, args.npatch, args.nlayer
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.te_dim-1)
        input_dim = 1 + args.te_dim
        self.ttcn_dim = args.hid_dim - 1
        self.Filter_Generators = nn.Sequential(
                nn.Linear(input_dim, self.ttcn_dim), nn.ReLU(inplace=True),
                nn.Linear(self.ttcn_dim, self.ttcn_dim), nn.ReLU(inplace=True),
                nn.Linear(self.ttcn_dim, input_dim*self.ttcn_dim))
        self.T_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))
        self.ADD_PE = PositionalEncoding(args.hid_dim)
        self.transformer_encoder = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hid_dim, nhead=args.nhead, batch_first=True), num_layers=args.tf_layer) for _ in range(self.n_layer)])
        self.supports_len = 1 if supports is None else len(supports) + 1
        self.supports = supports if supports is not None else []
        self.nodevec_dim = args.node_dim
        self.nodevec1 = nn.Parameter(torch.randn(self.N, self.nodevec_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(self.nodevec_dim, self.N), requires_grad=True)
        self.nodevec_linear1 = nn.ModuleList([nn.Linear(args.hid_dim, self.nodevec_dim) for _ in range(self.n_layer)])
        self.nodevec_linear2 = nn.ModuleList([nn.Linear(args.hid_dim, self.nodevec_dim) for _ in range(self.n_layer)])
        self.nodevec_gate1 = nn.ModuleList([nn.Sequential(nn.Linear(args.hid_dim+self.nodevec_dim, 1), nn.Tanh(), nn.ReLU()) for _ in range(self.n_layer)])
        self.nodevec_gate2 = nn.ModuleList([nn.Sequential(nn.Linear(args.hid_dim+self.nodevec_dim, 1), nn.Tanh(), nn.ReLU()) for _ in range(self.n_layer)])
        self.gconv = nn.ModuleList([gcn(args.hid_dim, args.hid_dim, dropout, support_len=self.supports_len, order=args.hop) for _ in range(self.n_layer)])
        self.temporal_agg = nn.Sequential(nn.Linear(args.hid_dim*self.M, args.hid_dim))
        self.decoder = nn.Sequential(
            nn.Linear(args.hid_dim + args.te_dim, args.hid_dim), nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim), nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, 1)
        )
    def LearnableTE(self, tt):
        return torch.cat([self.te_scale(tt), torch.sin(self.te_periodic(tt))], -1)
    def TTCN(self, X_int, mask_X):
        N_total, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2).view(N_total, Lx, self.ttcn_dim, -1)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1)
        return torch.relu(ttcn_out + self.T_bias)
    def IMTS_Model(self, x, mask_X, topo_pruning_mask=None):
        N_vars, M_patch = self.N, self.M
        B = x.shape[0] // (N_vars * M_patch)
        mask_patch_float = (mask_X.sum(dim=1) > 0).float()
        x_patch = self.TTCN(x, mask_X)
        x_patch = torch.cat([x_patch, mask_patch_float], dim=-1).view(B, N_vars, M_patch, -1)
        x = x_patch
        for layer in range(self.n_layer):
            if layer > 0: x_last = x.clone()
            x = x.reshape(B*N_vars, M_patch, -1)
            x = self.ADD_PE(x)
            if topo_pruning_mask is not None:
                pad_mask = (1 - topo_pruning_mask).bool().repeat_interleave(N_vars, dim=0)
                x = self.transformer_encoder[layer](x, src_key_padding_mask=pad_mask)
            else:
                x = self.transformer_encoder[layer](x)
            x = x.view(B, N_vars, M_patch, -1)
            nodevec1 = self.nodevec1.view(1, 1, N_vars, self.nodevec_dim).repeat(B, M_patch, 1, 1).to(x.device)
            nodevec2 = self.nodevec2.view(1, 1, self.nodevec_dim, N_vars).repeat(B, M_patch, 1, 1).to(x.device)
            x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
            x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))
            nodevec1 = nodevec1 + (x_gate1 * self.nodevec_linear1[layer](x)).permute(0,2,1,3)
            nodevec2 = nodevec2 + (x_gate2 * self.nodevec_linear2[layer](x)).permute(0,2,3,1)
            raw_adp = F.relu(torch.matmul(nodevec1, nodevec2))
            adp = F.softmax(raw_adp, dim=-1)
            new_supports = self.supports + [adp]
            x = self.gconv[layer](x.permute(0,3,1,2), new_supports).permute(0, 2, 3, 1)
            if layer > 0:
                if topo_pruning_mask is not None:
                    mask_expanded = topo_pruning_mask.unsqueeze(1).unsqueeze(-1)
                    x = (x_last + x) * mask_expanded
                else:
                    x = x_last + x
        return self.temporal_agg(x.reshape(B, N_vars, -1))
    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask, topo_pruning_mask=None):
        B, M, L_in, N = X.shape
        X_reshape = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        truth_ts_reshape = truth_time_steps.repeat(1, 1, 1, N).permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        mask_reshape = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        X_cat = torch.cat([X_reshape, self.LearnableTE(truth_ts_reshape)], dim=-1)
        h = self.IMTS_Model(X_cat, mask_reshape, topo_pruning_mask)
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
        decoder_input = torch.cat([h, self.LearnableTE(time_steps_to_predict)], dim=-1)
        return self.decoder(decoder_input).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)




PARAMS_41 = ['Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC']
PARAMS_DICT = {k: i for i, k in enumerate(PARAMS_41)}

def load_sparse_physionet(file_obj):
    try:
        df = pd.read_csv(file_obj, on_bad_lines='skip')
        if df.empty or len(df.columns) != 3: return None
        df.columns = ['Time', 'Parameter', 'Value']
        times, vals, masks = [], [], []
        grouped = df.groupby('Time')
        for time_str, group in grouped:
            try:
                h, m = map(int, str(time_str).split(':'))
                t_hours = h + m / 60.0
            except: continue
            val_vec, mask_vec = np.zeros(41), np.zeros(41)
            for param, sub_group in group.groupby('Parameter'):
                if param in PARAMS_DICT:
                    idx = PARAMS_DICT[param]
                    val_vec[idx] = sub_group['Value'].mean()
                    mask_vec[idx] = 1.0
            if mask_vec.sum() > 0:
                times.append(t_hours); vals.append(val_vec); masks.append(mask_vec)
        if len(times) == 0: return None
        sort_idx = np.argsort(times)
        return np.array(times)[sort_idx], np.array(vals)[sort_idx], np.array(masks)[sort_idx]
    except: return None

def build_patient_data(base_dir):
    patient_data = {}
    # The paper reports the 4,000-record PhysioNet set-a benchmark.  Do not
    # silently merge set-b/set-c, which would change the experimental cohort.
    tar_files = glob.glob(os.path.join(base_dir, '**/set-a.tar.gz'), recursive=True)
    if not tar_files:
        raise FileNotFoundError(
            "PhysioNet set-a.tar.gz was not found. Run "
            "`python scripts/download_data.py --dataset physionet`."
        )
    for tar_path in tar_files:
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.txt') and not os.path.basename(member.name).startswith('._'):
                    pid = os.path.basename(member.name).split('.')[0]
                    f_obj = tar.extractfile(member)
                    if f_obj is not None:
                        res = load_sparse_physionet(f_obj)
                        if res is not None: patient_data[pid] = res
    return patient_data

class PhysioNetSparseDataset(Dataset):
    def __init__(self, patient_dict, args):
        self.samples = []
        for pid in sorted(patient_dict.keys()):
            times, values, masks = patient_dict[pid]
            if len(np.where(times <= args.history)[0]) > 0 and len(np.where(times > args.history)[0]) > 0:
                self.samples.append({'times': times, 'values': values, 'masks': masks})
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def sparse_collate_fn_v31(batch, args, data_min, data_range, time_max, data_mean):
    """Dataset experiment entry point."""
    B, M, N = len(batch), args.npatch, args.ndim
    max_obs_len, max_pred_len, max_points = 0, 0, 24

    for s in batch:
        tt = s['times']
        for p in range(M):
            patch_end = p * args.stride + args.patch_size
            in_patch = (tt >= p * args.stride) & (
                (tt <= patch_end) if p == M - 1 else (tt < patch_end)
            )
            p_len = (in_patch & (tt <= args.history)).sum()
            max_obs_len = max(max_obs_len, min(p_len, max_points))
        max_pred_len = max(max_pred_len, (tt > args.history).sum())

    X_raw = torch.zeros(B, M, max_obs_len, N)
    truth_time_steps = torch.zeros(B, M, max_obs_len, 1)
    obs_mask = torch.zeros(B, M, max_obs_len, N)
    targets, target_mask = torch.zeros(B, max_pred_len, N), torch.zeros(B, max_pred_len, N)
    time_steps_to_predict = torch.zeros(B, max_pred_len)

    for i, s in enumerate(batch):
        tt, vals, m = s['times'], s['values'], s['masks']
        vals_n, tt_n = (vals - data_min) / data_range, tt / time_max
        for p in range(M):
            patch_end = p * args.stride + args.patch_size
            in_patch = (tt >= p * args.stride) & (
                (tt <= patch_end) if p == M - 1 else (tt < patch_end)
            )
            p_idx = np.where(in_patch & (tt <= args.history))[0]
            if len(p_idx) > max_points: p_idx = p_idx[np.linspace(0, len(p_idx)-1, max_points).astype(int)]
            if len(p_idx) > 0:
                X_raw[i, p, :len(p_idx), :] = torch.FloatTensor(vals_n[p_idx])
                truth_time_steps[i, p, :len(p_idx), 0] = torch.FloatTensor(tt_n[p_idx])
                obs_mask[i, p, :len(p_idx), :] = torch.FloatTensor(m[p_idx])
        pred_idx = np.where(tt > args.history)[0]
        if len(pred_idx) > 0:
            targets[i, :len(pred_idx), :] = torch.FloatTensor(vals_n[pred_idx])
            target_mask[i, :len(pred_idx), :] = torch.FloatTensor(m[pred_idx])
            time_steps_to_predict[i, :len(pred_idx)] = torch.FloatTensor(tt_n[pred_idx])

    return {'X_raw': X_raw, 'truth_time_steps': truth_time_steps, 'mask': obs_mask,
            'targets': targets, 'target_mask': target_mask,
            'time_steps_to_predict': time_steps_to_predict}



def get_sample_losses(pred, targets, target_mask):
    sq_error = ((targets - pred)**2) * target_mask
    valid = target_mask.sum(dim=(1, 2)) + 1e-8
    return sq_error.sum(dim=(1, 2)) / valid

def train_epoch(model, calib, loader, optimizer, device):
    model.train()
    calib.train()
    total_loss = 0
    for batch in loader:
        X_raw = batch['X_raw'].to(device)
        mask = batch['mask'].to(device)
        time_steps = batch['time_steps_to_predict'].to(device)
        truth_time_steps = batch['truth_time_steps'].to(device)
        targets = batch['targets'].to(device)
        target_masks = batch['target_mask'].to(device)

        X_cons, X_agg, tm_cons, tm_agg, taus, rhos, patch_scores, keep_cons, keep_agg = apply_addp_routing_calibrated(
            X_raw, mask, global_mean_for_calib, calib
        )

        X_concat = torch.cat([X_cons, X_agg], dim=0)
        tm_concat = torch.cat([tm_cons, tm_agg], dim=0)
        ts_concat = torch.cat([time_steps, time_steps], dim=0)
        truth_ts_concat = torch.cat([truth_time_steps, truth_time_steps], dim=0)
        mask_concat = torch.cat([mask, mask], dim=0)

        out_concat = model.forecasting(ts_concat, X_concat, truth_ts_concat, mask_concat, tm_concat).squeeze(0)
        B = targets.shape[0]
        out_cons, out_agg = out_concat[:B], out_concat[B:]

        loss_cons = get_sample_losses(out_cons, targets, target_masks)
        loss_agg = get_sample_losses(out_agg, targets, target_masks)
        stacked_losses = torch.stack([loss_cons, loss_agg], dim=1)
        taus = taus.unsqueeze(1)
        max_loss = stacked_losses.max(dim=1, keepdim=True)[0]
        dro_loss = taus * torch.log(torch.mean(torch.exp((stacked_losses - max_loss) / taus), dim=1, keepdim=True) + 1e-8) + max_loss
        auxiliary, _, _ = calibration_auxiliary_loss(
            patch_scores,
            keep_cons,
            rhos,
            score_weight=0.001,
        )
        final_loss = dro_loss.mean() + auxiliary

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(calib.parameters()), max_norm=5.0)
        optimizer.step()
        total_loss += final_loss.item()
    return total_loss / len(loader)

def evaluate_epoch(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            X_raw = batch['X_raw'].to(device)
            ts = batch['time_steps_to_predict'].to(device)
            truth_ts = batch['truth_time_steps'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)
            target_mask = batch['target_mask'].to(device)
            outputs = model.forecasting(ts, X_raw, truth_ts, mask, topo_pruning_mask=None)
            sample_mse = get_sample_losses(outputs.squeeze(0), targets, target_mask)
            total_mse += sample_mse.sum().item()
            total_samples += sample_mse.numel()
    return total_mse / max(total_samples, 1), 0




global_mean_for_calib = None

def main():
    print("=" * 70)
    print("ADDP - Learnable Calibrators - PhysioNet")
    print("ADDP with constrained learnable calibration")
    print("=" * 70)

    global global_mean_for_calib

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = os.environ.get('ADDP_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))
    base_dir = os.path.join(project_root, 'data', 'physionet')
    cache_path = os.path.join(base_dir, 'patient_data_cache.pkl')

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f: patient_data = pickle.load(f)
        if len(patient_data) != 4000:
            print("Ignoring a cache that does not contain the 4,000-record set-a cohort")
            patient_data = build_patient_data(base_dir)
            with open(cache_path, 'wb') as f: pickle.dump(patient_data, f)
    else:
        patient_data = build_patient_data(base_dir)
        with open(cache_path, 'wb') as f: pickle.dump(patient_data, f)

    args = Args(device=device)
    SEED = int(os.environ.get('ADDP_SEED', '0'))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    patient_ids = list(patient_data.keys())
    np.random.seed(42)
    np.random.shuffle(patient_ids)
    n_total = len(patient_ids)
    train_ids = patient_ids[:int(n_total*0.6)]
    val_ids = patient_ids[int(n_total*0.6):int(n_total*0.8)]
    test_ids = patient_ids[int(n_total*0.8):]

    train_dict = {pid: patient_data[pid] for pid in train_ids}
    all_vals, all_masks = [], []
    for pid, (times, values, masks) in train_dict.items():
        if (times <= args.history).any():
            all_vals.append(values[times <= args.history])
            all_masks.append(masks[times <= args.history])
    masked_vals = np.where(np.concatenate(all_masks, axis=0) == 1.0, np.concatenate(all_vals, axis=0), np.nan)
    with np.errstate(all='ignore'):
        data_min = np.nan_to_num(np.nanmin(masked_vals, axis=0), nan=0.0)
        data_max = np.nan_to_num(np.nanmax(masked_vals, axis=0), nan=1.0)
    data_range = np.where((data_max - data_min) == 0, 1.0, data_max - data_min)
    data_mean = torch.FloatTensor(np.nan_to_num(np.nanmean((masked_vals - data_min) / data_range, axis=0), nan=0.0))
    global_mean_for_calib = data_mean.to(device)

    print(f"Dimensions: {args.ndim}, patches: {args.npatch}")
    print(f"Samples: train={len(train_ids)}, validation={len(val_ids)}, test={len(test_ids)}")

    train_loader = DataLoader(PhysioNetSparseDataset(train_dict, args), batch_size=32, shuffle=True, collate_fn=lambda b: sparse_collate_fn_v31(b, args, data_min, data_range, 48.0, data_mean))
    val_loader = DataLoader(PhysioNetSparseDataset({pid: patient_data[pid] for pid in val_ids}, args), batch_size=32, collate_fn=lambda b: sparse_collate_fn_v31(b, args, data_min, data_range, 48.0, data_mean))
    test_loader = DataLoader(PhysioNetSparseDataset({pid: patient_data[pid] for pid in test_ids}, args), batch_size=32, collate_fn=lambda b: sparse_collate_fn_v31(b, args, data_min, data_range, 48.0, data_mean))

    model = ADDPModel(args).to(device)
    calib = ADDPCalibration().to(device)

    print("\nInitial calibration")
    desc_init = calib.describe()
    for k, v in desc_init.items():
        if isinstance(v, list):
            print(f"  {k}: {['%.4f' % x for x in v]}")
        else:
            print(f"  {k}: {v:.4f}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(calib.parameters()),
        lr=0.001
    )

    best_val_loss = float('inf')
    best_model_state = None
    best_calib_state = None
    best_epoch = 0
    save_path = f'physionet_addp_seed{SEED}_best.pth'
    epochs, patience, no_improve_cnt = 50, 10, 0

    print(f"\nTraining for at most {epochs} epochs (patience={patience})")
    for epoch in range(epochs):
        train_loss = train_epoch(model, calib, train_loader, optimizer, device)
        val_mse, _ = evaluate_epoch(model, val_loader, device)

        print(f"Epoch {epoch:2d}: Train DRO = {train_loss:.5f}, Val MSE = {val_mse:.5f}")
        if not np.isnan(val_mse) and val_mse < best_val_loss:
            best_val_loss, no_improve_cnt = val_mse, 0
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_calib_state = {k: v.cpu().clone() for k, v in calib.state_dict().items()}
            torch.save(
                {'model': best_model_state, 'calibration': best_calib_state,
                 'seed': SEED, 'best_epoch': best_epoch},
                save_path,
            )
            print(f"  Saved best checkpoint (validation MSE: {val_mse:.6f})")
        else:
            no_improve_cnt += 1
        if no_improve_cnt >= patience:
            print("Early stopping")
            break

    print(f"\nBest epoch: {best_epoch}; validation MSE: {best_val_loss:.6f}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        calib.load_state_dict(best_calib_state)

    test_mse, _ = evaluate_epoch(model, test_loader, device)

    print("\n" + "=" * 60)
    print("\nTest results")
    print(f"  Test MSE : {test_mse:.6f}")
    print("=" * 60)

    print("\nLearned calibration")
    desc_final = calib.describe()
    for k, v in desc_final.items():
        if isinstance(v, list):
            print(f"  {k}: {['%.4f' % x for x in v]}")
        else:
            print(f"  {k}: {v:.4f}")

    print("\nMonotonicity checks")
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    rhos = [desc_final['rho_curve'][i] for i in range(5)]
    taus = [desc_final['tau_curve'][i] for i in range(5)]
    rho_mono = all(rhos[i] >= rhos[i+1] for i in range(4))
    tau_mono = all(taus[i] >= taus[i+1] for i in range(4))
    print(f"  core ratio decreases with gamma: {rho_mono}")
    print(f"  temperature decreases with gamma: {tau_mono}")
    print(f"  core ratio range: [{min(rhos):.4f}, {max(rhos):.4f}] (init: [{min(desc_init['rho_curve']):.4f}, {max(desc_init['rho_curve']):.4f}])")
    print(f"  temperature range: [{min(taus):.4f}, {max(taus):.4f}] (init: [{min(desc_init['tau_curve']):.4f}, {max(desc_init['tau_curve']):.4f}])")
    print(f"  high weights init=[{desc_init['alpha_h'][0]:.3f},{desc_init['alpha_h'][1]:.3f}] -> final=[{desc_final['alpha_h'][0]:.4f}, {desc_final['alpha_h'][1]:.4f}]")
    print(f"  low weights init=[{desc_init['alpha_l'][0]:.3f},{desc_init['alpha_l'][1]:.3f},{desc_init['alpha_l'][2]:.3f}] -> final=[{desc_final['alpha_l'][0]:.4f}, {desc_final['alpha_l'][1]:.4f}, {desc_final['alpha_l'][2]:.4f}]")
    print(f"  recency weight init={desc_init['lambda_u']:.4f} -> final={desc_final['lambda_u']:.4f}")
    print(f"  core slope: {desc_init['b_c']:.4f} -> {desc_final['b_c']:.4f}")
    print(f"  temperature slope: {desc_init['b_tau']:.4f} -> {desc_final['b_tau']:.4f}")

if __name__ == "__main__":
    main()
