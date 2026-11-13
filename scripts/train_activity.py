"""Dataset experiment entry point."""
import os
import json
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from addp import (
    ADDPCalibration,
    apply_addp_routing_calibrated,
    calibration_auxiliary_loss,
)


class Args:
    def __init__(self, device, n_patch, pred_len, ndim=12):
        self.device = device
        self.hid_dim = 32
        self.ndim = ndim
        self.npatch = n_patch
        self.nlayer = 1
        self.te_dim = 10
        self.node_dim = 10
        self.nhead = 2
        self.tf_layer = 1
        self.hop = 1
        self.outlayer = "Linear"
        self.pred_len = pred_len



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
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.order = order
    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
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
    """Dataset experiment entry point."""
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
            nn.Linear(args.hid_dim + args.te_dim if use_future_timestamps else args.hid_dim, args.hid_dim), nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim), nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, 1)
        )

    def LearnableTE(self, tt):
        return torch.cat([self.te_scale(tt), torch.sin(self.te_periodic(tt))], -1)

    def TTCN(self, X_int, mask_X):
        N_total, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
        attn = F.softmax(Filter_mask, dim=-2)
        attn = attn * mask_X
        attn = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)
        Filter_seqnorm = attn.view(N_total, Lx, self.ttcn_dim, -1)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        return torch.relu(torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) + self.T_bias)

    def IMTS_Model(self, x, mask_X, topo_pruning_mask=None):
        N_vars, M_patch = self.N, self.M
        B = x.shape[0] // (N_vars * M_patch)

        transformer_mask = (mask_X.sum(dim=1) == 0).view(B * N_vars, M_patch)
        all_empty_seqs = transformer_mask.all(dim=1)
        transformer_mask[all_empty_seqs, 0] = False

        if topo_pruning_mask is not None:
            topo_ignore = (1 - topo_pruning_mask).bool().repeat_interleave(N_vars, dim=0)
            transformer_mask = transformer_mask | topo_ignore

        true_empty_mask = transformer_mask.view(B, N_vars, M_patch).permute(0, 2, 1)

        mask_patch_float = (mask_X.sum(dim=1) > 0).float()
        x_patch = self.TTCN(x, mask_X) * mask_patch_float

        x_patch = torch.cat([x_patch, mask_patch_float], dim=-1).view(B, N_vars, M_patch, -1)
        x = x_patch

        for layer in range(self.n_layer):
            if layer > 0: x_last = x.clone()
            x = x.reshape(B*N_vars, M_patch, -1)
            x = self.ADD_PE(x)

            x = self.transformer_encoder[layer](x, src_key_padding_mask=transformer_mask)
            x = x * (~all_empty_seqs).view(B * N_vars, 1, 1).float()
            x = x.view(x_patch.shape)

            nodevec1 = self.nodevec1.view(1, 1, N_vars, self.nodevec_dim).repeat(B, M_patch, 1, 1).to(x.device)
            nodevec2 = self.nodevec2.view(1, 1, self.nodevec_dim, N_vars).repeat(B, M_patch, 1, 1).to(x.device)
            x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
            x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))
            nodevec1 = nodevec1 + (x_gate1 * self.nodevec_linear1[layer](x)).permute(0,2,1,3)
            nodevec2 = nodevec2 + (x_gate2 * self.nodevec_linear2[layer](x)).permute(0,3,2,1)

            valid_nodes = (~true_empty_mask).float()
            gnn_mask = valid_nodes.unsqueeze(-1) * valid_nodes.unsqueeze(-2)

            raw_adp = F.relu(torch.matmul(nodevec1, nodevec2))
            raw_adp = raw_adp.masked_fill(gnn_mask == 0, -1e9)
            adp = F.softmax(raw_adp, dim=-1).masked_fill(torch.isnan(F.softmax(raw_adp, dim=-1)), 0.0)

            new_supports = self.supports + [adp]
            x = self.gconv[layer](x.permute(0,3,1,2), new_supports).permute(0, 2, 3, 1)

            if layer > 0: x = x_last + x

        return self.temporal_agg(x.reshape(B, N_vars, -1))

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask, topo_pruning_mask=None):
        B, M, L_in, N = X.shape
        X = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        if truth_time_steps.shape[-1] == 1: truth_time_steps = truth_time_steps.expand(-1, -1, -1, N)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        mask = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)

        X = torch.cat([X, self.LearnableTE(truth_time_steps)], dim=-1)
        h = self.IMTS_Model(X, mask, topo_pruning_mask)

        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)

        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
        decoder_input = torch.cat([h, self.LearnableTE(time_steps_to_predict)], dim=-1)

        return self.decoder(decoder_input).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)


# ============ 3. Learnable Calibrators ============

def load_conf_long_data(data_path):
    columns = ['subject', 'sensor', 'timestamp', 'datetime', 'acc_x', 'acc_y', 'acc_z', 'activity']
    df = pd.read_csv(data_path, header=None, names=columns)
    valid_sensors = ["010-000-024-033", "010-000-030-096", "020-000-033-111", "020-000-032-221"]
    df = df[df['sensor'].isin(valid_sensors)].copy()
    timestamp_ms = df['timestamp'] / 10000.0
    df['timestamp_ms'] = (np.floor(timestamp_ms / 100.0) * 100.0).astype(np.int64)
    axes = ['acc_x', 'acc_y', 'acc_z']
    pivot = df.pivot_table(
        index=['subject', 'timestamp_ms'], columns='sensor', values=axes, aggfunc='mean'
    )
    expected = pd.MultiIndex.from_product([axes, valid_sensors])
    pivot = pivot.reindex(columns=expected)
    observed = pivot.notna().astype(np.float32)
    pivot = pivot.fillna(0.0).astype(np.float32)
    value_columns = [f'{axis}_{sensor}' for axis, sensor in expected]
    mask_columns = [f'mask_{column}' for column in value_columns]
    pivot.columns = value_columns
    observed.columns = mask_columns
    return pd.concat([pivot, observed], axis=1).reset_index()

def generate_instances(df, chunk_len_ms=4000, obs_window_ms=3000, pred_window_ms=1000, stride_ms=500):
    instances = []
    subjects = df['subject'].unique()
    feature_cols = [c for c in df.columns if c not in ['subject', 'timestamp_ms'] and not c.startswith('mask_')]
    mask_cols = [f'mask_{column}' for column in feature_cols]

    for subject in subjects:
        sub_df = df[df['subject'] == subject].copy()
        sub_df = sub_df.sort_values('timestamp_ms')
        timestamps = sub_df['timestamp_ms'].values
        values = sub_df[feature_cols].values
        observation_masks = sub_df[mask_cols].values

        t_min, t_max = timestamps.min(), timestamps.max()
        start = t_min
        while start + chunk_len_ms <= t_max:
            end = start + chunk_len_ms
            mask = (timestamps >= start) & (timestamps < end)
            chunk_ts = timestamps[mask] - start
            chunk_vals = values[mask]
            chunk_masks = observation_masks[mask]

            if len(chunk_ts) > 0:
                instances.append({
                    'timestamps': chunk_ts, 'values': chunk_vals, 'masks': chunk_masks,
                    'obs_end': obs_window_ms, 'pred_start': obs_window_ms, 'pred_end': obs_window_ms + pred_window_ms
                })
            start += stride_ms
    return instances

def compute_minmax_params(instances):
    variable_count = instances[0]['values'].shape[1]
    observed_values = [[] for _ in range(variable_count)]
    for inst in instances:
        history = inst['timestamps'] < inst['obs_end']
        for variable in range(variable_count):
            valid = history & (inst['masks'][:, variable] > 0)
            if valid.any():
                observed_values[variable].append(inst['values'][valid, variable])
    d_min = np.zeros(variable_count, dtype=np.float32)
    d_max = np.ones(variable_count, dtype=np.float32)
    normalized_mean = np.zeros(variable_count, dtype=np.float32)
    for variable, parts in enumerate(observed_values):
        if not parts:
            continue
        values = np.concatenate(parts)
        d_min[variable] = values.min()
        d_max[variable] = values.max()
        scale = max(float(d_max[variable] - d_min[variable]), 1e-8)
        normalized_mean[variable] = ((values - d_min[variable]) / scale).mean()
    scale = d_max - d_min
    scale[scale == 0] = 1.0
    return d_min, scale, normalized_mean

class ActivityDatasetADDP(Dataset):
    def __init__(self, instances, patch_window_ms=300, pred_window_ms=1000, obs_window_ms=3000, n_patch=10, pred_len=5, d_min=None, scale=None, max_points=24):
        self.d_min = d_min
        self.scale = scale
        self.N_dim = instances[0]['values'].shape[1]
        self.samples = []

        for inst in instances:
            timestamps = inst['timestamps']
            values = inst['values']
            observation_masks = inst['masks']
            patches, patch_masks, patch_times = [], [], []

            for p in range(n_patch):
                patch_start, patch_end = p * patch_window_ms, (p + 1) * patch_window_ms
                indices = np.where((timestamps >= patch_start) & (timestamps < patch_end))[0]

                is_valid_patch = False
                if len(indices) > 0:
                    patch_vals = values[indices]
                    patch_mask = observation_masks[indices]
                    patch_ts = timestamps[indices] - patch_start
                    is_valid_patch = True

                    if len(patch_vals) > max_points:
                        uniform_idx = np.round(np.linspace(0, len(patch_vals) - 1, max_points)).astype(int)
                        patch_vals = patch_vals[uniform_idx]
                        patch_mask = patch_mask[uniform_idx]
                        patch_ts = patch_ts[uniform_idx]

                    patch_ts_norm = patch_ts / patch_window_ms

                if not is_valid_patch:
                    patch_vals = np.zeros((1, self.N_dim), dtype=np.float32)
                    patch_mask = np.zeros((1, self.N_dim), dtype=np.float32)
                    patch_ts_norm = np.array([0.5], dtype=np.float32)

                patches.append(patch_vals.astype(np.float32))
                patch_times.append(patch_ts_norm.astype(np.float32))
                patch_masks.append(patch_mask.astype(np.float32))

            pred_start, pred_end = inst['pred_start'], inst['pred_end']
            future_times_rel = np.linspace(0, 1, pred_len).astype(np.float32)
            future_times_abs = pred_start + future_times_rel * (pred_end - pred_start)

            pred_indices = np.where((timestamps >= pred_start) & (timestamps < pred_end))[0]
            target_values = np.zeros((pred_len, self.N_dim), dtype=np.float32)
            target_masks = np.zeros((pred_len, self.N_dim), dtype=np.float32)
            for i, ft_abs in enumerate(future_times_abs):
                if len(pred_indices) > 0:
                    idx = pred_indices[np.argmin(np.abs(timestamps[pred_indices] - ft_abs))]
                    target_values[i] = values[idx]
                    target_masks[i] = observation_masks[idx]

            self.samples.append({
                'patches': patches, 'patch_times': patch_times, 'patch_masks': patch_masks,
                'target_values': target_values, 'target_masks': target_masks,
                'future_times_rel': future_times_rel
            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        M, N = len(sample['patches']), self.N_dim
        max_len = max(len(p) for p in sample['patches'])

        X = torch.zeros(M, max_len, N)
        truth_time_steps = torch.zeros(M, max_len, 1)
        mask = torch.zeros(M, max_len, N)

        for p in range(M):
            patch_vals = sample['patches'][p]
            patch_ts = sample['patch_times'][p]
            patch_mask = sample['patch_masks'][p]
            L = len(patch_vals)
            if self.d_min is not None and self.scale is not None:
                patch_vals = (patch_vals - self.d_min) / self.scale

            X[p, :L, :] = torch.FloatTensor(patch_vals)
            truth_time_steps[p, :L, 0] = torch.FloatTensor(patch_ts)
            mask[p, :L, :] = torch.FloatTensor(patch_mask)

        target_vals = sample['target_values']
        if self.d_min is not None:
            target_vals = (target_vals - self.d_min) / self.scale

        return {'X_raw': X, 'truth_time_steps': truth_time_steps, 'mask': mask,
                'target_values': torch.FloatTensor(target_vals),
                'target_masks': torch.FloatTensor(sample['target_masks']),
                'future_times_rel': torch.FloatTensor(sample['future_times_rel'])}


def create_collate_fn_v31(global_mean):
    """Dataset experiment entry point."""
    def collate_fn(batch):
        B = len(batch)
        M = batch[0]['X_raw'].shape[0]
        max_len = max(b['X_raw'].shape[1] for b in batch)
        N = batch[0]['X_raw'].shape[-1]
        pred_len = batch[0]['target_values'].shape[0]

        X_raw = torch.zeros(B, M, max_len, N)
        truth_time_steps = torch.zeros(B, M, max_len, 1)
        mask = torch.zeros(B, M, max_len, N)
        targets = torch.zeros(B, pred_len, N)
        target_masks = torch.zeros(B, pred_len, N)
        future_times_rel = torch.zeros(B, pred_len)

        for i, sample in enumerate(batch):
            x = sample['X_raw']
            L = x.shape[1]
            X_raw[i, :, :L, :] = x
            truth_time_steps[i, :, :L, :] = sample['truth_time_steps']
            mask[i, :, :L, :] = sample['mask']
            targets[i] = sample['target_values']
            target_masks[i] = sample['target_masks']
            future_times_rel[i] = sample['future_times_rel']

        return {
            'X_raw': X_raw, 'truth_time_steps': truth_time_steps,
            'mask': mask, 'targets': targets, 'target_masks': target_masks,
            'time_steps_to_predict': future_times_rel,
            'global_mean': torch.FloatTensor(global_mean),
        }
    return collate_fn



def get_sample_losses(pred, targets, target_masks):
    sq_error = ((targets - pred)**2) * target_masks
    valid = target_masks.sum(dim=(1, 2)) + 1e-8
    return sq_error.sum(dim=(1, 2)) / valid

def compute_macro_loss_components(pred, targets, masks):
    N_dim = targets.shape[-1]
    sq_error = ((pred - targets) ** 2) * masks
    abs_error = torch.abs(pred - targets) * masks
    error_var_sum = sq_error.reshape(-1, N_dim).sum(dim=0)
    mae_var_sum = abs_error.reshape(-1, N_dim).sum(dim=0)
    mask_count = masks.reshape(-1, N_dim).sum(dim=0)
    return error_var_sum, mae_var_sum, mask_count

def macro_loss(error_var_sum, mask_count):
    error_var_avg = error_var_sum / (mask_count + 1e-8)
    n_avai_var = torch.count_nonzero(mask_count)
    if n_avai_var == 0: return torch.tensor(0.0).to(error_var_sum.device)
    return error_var_avg.sum() / n_avai_var

def train_epoch(model, calib, dataloader, optimizer, device):
    model.train()
    calib.train()
    total_loss = 0
    for batch in dataloader:
        X_raw = batch['X_raw'].to(device)
        mask = batch['mask'].to(device)
        global_mean = batch['global_mean'].to(device)
        time_steps = batch['time_steps_to_predict'].to(device)
        truth_time_steps = batch['truth_time_steps'].to(device)
        targets = batch['targets'].to(device)
        target_masks = batch['target_masks'].to(device)

        # routing (calibrated, autograd)
        X_cons, X_agg, tm_cons, tm_agg, taus, rhos, patch_scores, keep_cons, keep_agg = apply_addp_routing_calibrated(
            X_raw, mask, global_mean, calib
        )

        # Concat Trick
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
            score_weight=0.01,
        )
        final_loss = dro_loss.mean() + auxiliary

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(calib.parameters()), max_norm=5.0)
        optimizer.step()
        total_loss += final_loss.item()
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_mse_sum = torch.zeros(model.N).to(device)
    total_mae_sum = torch.zeros(model.N).to(device)
    total_mask_count = torch.zeros(model.N).to(device)

    with torch.no_grad():
        for batch in dataloader:
            X_raw, time_steps = batch['X_raw'].to(device), batch['time_steps_to_predict'].to(device)
            truth_time_steps, mask = batch['truth_time_steps'].to(device), batch['mask'].to(device)
            targets, target_masks = batch['targets'].to(device), batch['target_masks'].to(device)

            outputs = model.forecasting(time_steps, X_raw, truth_time_steps, mask, topo_pruning_mask=None)
            pred = outputs.squeeze(0)

            e_sum, mae_sum, m_count = compute_macro_loss_components(pred, targets, target_masks)
            total_mse_sum += e_sum
            total_mae_sum += mae_sum
            total_mask_count += m_count

    final_mse = macro_loss(total_mse_sum, total_mask_count).item()
    final_mae = macro_loss(total_mae_sum, total_mask_count).item()
    return final_mse, final_mae, math.sqrt(final_mse)



def main():
    print("=" * 70)
    print("ADDP - Learnable Calibrators - Activity")
    print("ADDP with constrained learnable calibration")
    print("Shared roughness routing with monotone constrained mappings")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.environ.get('ADDP_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))
    data_path = os.path.join(base_dir, 'data', 'activity', 'ConfLongDemo_JSI.txt')

    chunk_len_ms, obs_window_ms, pred_window_ms, patch_window_ms, stride_ms = 4000, 3000, 1000, 300, 500
    n_patch = obs_window_ms // patch_window_ms
    pred_len, batch_size, epochs, learning_rate = 5, 32, 50, 0.001
    SEED = int(os.environ.get('ADDP_SEED', '0'))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    df = load_conf_long_data(data_path)
    split_path = os.path.join(
        project_root, 'data', 'activity', 'recording_split.json'
    )
    with open(split_path, 'r', encoding='utf-8') as handle:
        split = json.load(handle)
    train_subjects = np.asarray(split['train'])
    val_subjects = np.asarray(split['validation'])
    test_subjects = np.asarray(split['test'])
    available = set(df['subject'].unique())
    declared = set(train_subjects) | set(val_subjects) | set(test_subjects)
    if declared != available:
        raise ValueError(
            "Activity recording_split.json does not match the recording IDs "
            "present in ConfLongDemo_JSI.txt."
        )
    assert not (set(train_subjects) & set(val_subjects))
    assert not (set(train_subjects) & set(test_subjects))
    assert not (set(val_subjects) & set(test_subjects))
    print(f"Recording-ID split: train={train_subjects.tolist()}")
    print(f"Recording-ID split: validation={val_subjects.tolist()}")
    print(f"Recording-ID split: test={test_subjects.tolist()}")

    train_instances = generate_instances(
        df[df['subject'].isin(train_subjects)].copy(),
        chunk_len_ms, obs_window_ms, pred_window_ms, stride_ms,
    )
    val_instances = generate_instances(
        df[df['subject'].isin(val_subjects)].copy(),
        chunk_len_ms, obs_window_ms, pred_window_ms, stride_ms,
    )
    test_instances = generate_instances(
        df[df['subject'].isin(test_subjects)].copy(),
        chunk_len_ms, obs_window_ms, pred_window_ms, stride_ms,
    )

    train_min, train_scale, train_mean = compute_minmax_params(train_instances)
    dynamic_ndim = train_instances[0]['values'].shape[1]

    print(f"Dimensions: {dynamic_ndim}")
    print(f"Split: train={len(train_instances)}, validation={len(val_instances)}, test={len(test_instances)}")

    train_dataset = ActivityDatasetADDP(train_instances, patch_window_ms, pred_window_ms, obs_window_ms, n_patch, pred_len, d_min=train_min, scale=train_scale)
    val_dataset = ActivityDatasetADDP(val_instances, patch_window_ms, pred_window_ms, obs_window_ms, n_patch, pred_len, d_min=train_min, scale=train_scale)
    test_dataset = ActivityDatasetADDP(test_instances, patch_window_ms, pred_window_ms, obs_window_ms, n_patch, pred_len, d_min=train_min, scale=train_scale)

    collate_fn = create_collate_fn_v31(train_mean)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    args = Args(device=device, n_patch=n_patch, pred_len=pred_len, ndim=dynamic_ndim)
    model = ADDPModel(args, use_future_timestamps=True).to(device)
    calib = ADDPCalibration().to(device)

    print("\nInitial calibration")
    desc_init = calib.describe()
    for k, v in desc_init.items():
        if isinstance(v, list):
            print(f"  {k}: {['%.4f'%x for x in v]}")
        else:
            print(f"  {k}: {v:.4f}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(calib.parameters()),
        lr=learning_rate
    )

    best_val_loss = float('inf')
    best_model_state = None
    best_calib_state = None
    best_epoch = 0
    save_path = f'activity_addp_seed{SEED}_best.pth'

    print(f"\nTraining for {epochs} epochs")
    for epoch in range(epochs):
        train_loss = train_epoch(model, calib, train_loader, optimizer, device)
        val_mse, val_mae, _ = evaluate_epoch(model, val_loader, device)

        print(f"Epoch {epoch:2d}: Train DRO = {train_loss:.6f}, Val MSE = {val_mse:.6f}")
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_calib_state = {k: v.cpu().clone() for k, v in calib.state_dict().items()}
            torch.save(
                {'model': best_model_state, 'calibration': best_calib_state,
                 'seed': SEED, 'best_epoch': best_epoch},
                save_path,
            )
            print(f"  Saved best checkpoint (validation MSE: {val_mse:.6f})")

    print(f"\nBest epoch: {best_epoch}; validation MSE: {best_val_loss:.6f}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        calib.load_state_dict(best_calib_state)

    test_mse, test_mae, test_rmse = evaluate_epoch(model, test_loader, device)

    print("\n" + "=" * 60)
    print("\nTest results")
    print(f"  Test MSE  : {test_mse:.6f}")
    print(f"  Test MAE  : {test_mae:.6f}")
    print(f"  Test RMSE : {test_rmse:.6f}")
    print("=" * 60)

    print("\nLearned calibration")
    desc_final = calib.describe()
    for k, v in desc_final.items():
        if isinstance(v, list):
            print(f"  {k}: {['%.4f'%x for x in v]}")
        else:
            print(f"  {k}: {v:.4f}")


    print("\nMonotonicity checks")
    init_rho = desc_init['rho_curve']
    final_rho = desc_final['rho_curve']
    init_tau = desc_init['tau_curve']
    final_tau = desc_final['tau_curve']
    rho_mono_ok = all(final_rho[i] >= final_rho[i+1] for i in range(len(final_rho)-1))
    tau_mono_ok = all(final_tau[i] >= final_tau[i+1] for i in range(len(final_tau)-1))
    print(f"  core ratio decreases with gamma: {rho_mono_ok}")
    print(f"  temperature decreases with gamma: {tau_mono_ok}")
    print(f"  core ratio range: [{min(final_rho):.4f}, {max(final_rho):.4f}] "
          f"(init: [{min(init_rho):.4f}, {max(init_rho):.4f}])")
    print(f"  temperature range: [{min(final_tau):.4f}, {max(final_tau):.4f}] "
          f"(init: [{min(init_tau):.4f}, {max(init_tau):.4f}])")
    print(f"  high weights init=[0.625,0.375] -> final={['%.4f'%x for x in desc_final['alpha_h']]}")
    print(f"  low weights init=[0.4,0.4,0.2] -> final={['%.4f'%x for x in desc_final['alpha_l']]}")
    print(f"  recency weight init=0.3000 -> final={desc_final['lambda_u']:.4f}")
    print(f"  core slope: {desc_init['b_c']:.4f} -> {desc_final['b_c']:.4f}")
    print(f"  temperature slope: {desc_init['b_tau']:.4f} -> {desc_final['b_tau']:.4f}")


if __name__ == "__main__":
    main()
