import os
from pathlib import Path
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import copy
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from addp import (
    ADDPCalibration,
    apply_addp_routing_calibrated,
    calibration_auxiliary_loss,
)


class Args:
    def __init__(self, device):
        self.device = device
        self.ndim = 96
        self.hid_dim = 64
        self.nlayer = 2
        self.nhead = 4
        self.tf_layer = 2
        self.hop = 2
        self.epoch = 50
        self.batch_size = 8
        self.lr = 2.5e-3
        self.history = 24.0
        self.pred_window = 24.0
        self.patch_size = 4.0
        self.stride = 4.0
        self.time_max = 48.0
        self.npatch = int(np.ceil((self.history - self.patch_size) / self.stride)) + 1
        self.te_dim = 10
        self.node_dim = 10
        self.outlayer = "Linear"
        self.max_points = 50


class nconv(nn.Module):
    def __init__(self): super(nconv, self).__init__()
    def forward(self, x, A): return torch.einsum('bfnm,bmnv->bfvm', (x, A)).contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
    def forward(self, x): return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.order = order
    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a); out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a); out.append(x2); x1 = x2
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
        self.te_periodic = nn.Linear(1, args.te_dim - 1)
        input_dim = 1 + args.te_dim
        self.ttcn_dim = args.hid_dim - 1
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, self.ttcn_dim), nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.ttcn_dim), nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, input_dim*self.ttcn_dim))
        self.T_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))
        self.ADD_PE = PositionalEncoding(args.hid_dim)
        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=args.hid_dim, nhead=args.nhead, batch_first=True),
                num_layers=args.tf_layer, enable_nested_tensor=False)
            for _ in range(self.n_layer)])
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
            nn.Linear(args.hid_dim, 1))
    def LearnableTE(self, tt):
        return torch.cat([self.te_scale(tt), torch.sin(self.te_periodic(tt))], -1)
    def TTCN(self, X_int, mask_X):
        N_total, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
        attn = F.softmax(Filter_mask, dim=-2) * mask_X
        attn = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)
        Filter_seqnorm = attn.reshape(N_total, Lx, self.ttcn_dim, -1)
        X_int_unsq = X_int.unsqueeze(dim=-2)
        return torch.relu(torch.sum(torch.sum(X_int_unsq * Filter_seqnorm, dim=-3), dim=-1) + self.T_bias)
    def IMTS_Model(self, x, mask_X, topo_pruning_mask, B, M, L_in):
        N_vars = self.N
        mask_patch = mask_X.reshape(B * N_vars, M, L_in).sum(dim=2) > 0
        mask_patch_float = mask_patch.float()
        transformer_mask = ~mask_patch
        if topo_pruning_mask is not None:
            topo_ignore = (topo_pruning_mask < 0.5).repeat_interleave(N_vars, dim=0)
            transformer_mask = transformer_mask | topo_ignore
        all_empty_seqs = transformer_mask.all(dim=1)
        transformer_mask[all_empty_seqs, 0] = False
        true_empty_mask = transformer_mask.reshape(B, N_vars, M).permute(0, 2, 1)
        x_reshaped = x.reshape(B * N_vars, M, L_in, -1)
        mask_reshaped = mask_X.reshape(B * N_vars, M, L_in, -1)
        x_patch_list = []
        for p in range(M):
            x_p = x_reshaped[:, p, :, :]
            mask_p = mask_reshaped[:, p, :, :]
            x_patch_list.append(self.TTCN(x_p, mask_p))
        x_patch = torch.stack(x_patch_list, dim=1)
        x_patch = x_patch * mask_patch_float.unsqueeze(-1)
        x_patch = torch.cat([x_patch, mask_patch_float.unsqueeze(-1)], dim=-1)
        x = x_patch.reshape(B, N_vars, M, -1).clone()
        for layer in range(self.n_layer):
            if layer > 0: x_last = x.clone()
            x = x.reshape(B * N_vars, M, -1)
            x = self.ADD_PE(x)
            x = self.transformer_encoder[layer](x, src_key_padding_mask=transformer_mask)
            x = x.reshape(B, N_vars, M, -1)
            nodevec1 = self.nodevec1.reshape(1, 1, N_vars, self.nodevec_dim).repeat(B, M, 1, 1).to(x.device)
            nodevec2 = self.nodevec2.reshape(1, 1, self.nodevec_dim, N_vars).repeat(B, M, 1, 1).to(x.device)
            x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
            x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))
            nodevec1 = nodevec1 + (x_gate1 * self.nodevec_linear1[layer](x)).permute(0, 2, 1, 3)
            nodevec2 = nodevec2 + (x_gate2 * self.nodevec_linear2[layer](x)).permute(0, 2, 3, 1)
            valid_nodes = (~true_empty_mask).float()
            gnn_mask = valid_nodes.unsqueeze(-1) * valid_nodes.unsqueeze(-2)
            raw_adp = F.relu(torch.matmul(nodevec1, nodevec2))
            raw_adp = raw_adp.masked_fill(gnn_mask == 0, -1e9)
            adp = torch.nan_to_num(F.softmax(raw_adp, dim=-1), nan=0.0)
            new_supports = self.supports + [adp]
            x = self.gconv[layer](x.permute(0, 3, 1, 2), new_supports).permute(0, 2, 3, 1)
            if layer > 0: x = x_last + x
            x = x * mask_patch_float.reshape(B, N_vars, M, 1)
        return self.temporal_agg(x.reshape(B, N_vars, -1))
    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask, topo_pruning_mask=None):
        B, M, L_in, N = X.shape
        X_flat = X.permute(0, 3, 1, 2).reshape(B * N, M * L_in, 1)
        if truth_time_steps.shape[-1] == 1:
            truth_time_steps = truth_time_steps.expand(-1, -1, -1, N)
        ts_flat = truth_time_steps.permute(0, 3, 1, 2).reshape(B * N, M * L_in, 1)
        mask_flat = mask.permute(0, 3, 1, 2).reshape(B * N, M * L_in, 1)
        X_combined = torch.cat([X_flat, self.LearnableTE(ts_flat)], dim=-1)
        h = self.IMTS_Model(X_combined, mask_flat, topo_pruning_mask, B, M, L_in)
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)
        if self.use_future_timestamps:
            p_ts = time_steps_to_predict.reshape(B, 1, L_pred, 1).repeat(1, N, 1, 1)
            te_pred = self.LearnableTE(p_ts)
            h = torch.cat([h, te_pred], dim=-1)
        return self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)




class MimicDataset(Dataset):
    def __init__(self, data_list): self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_data_min_max_and_mean(data_list, ndim=96):
    observed = [[] for _ in range(ndim)]
    for (_, times, vals, mask) in data_list:
        history = times < 24.0
        for variable in range(ndim):
            valid = history & (mask[:, variable] > 0)
            if valid.any():
                observed[variable].append(vals[valid, variable].float())

    d_min = torch.zeros(ndim)
    d_max = torch.ones(ndim)
    global_mean = torch.zeros(ndim)
    for variable, parts in enumerate(observed):
        if not parts:
            continue
        values = torch.cat(parts)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            continue
        d_min[variable] = values.min()
        d_max[variable] = values.max()
        scale = (d_max[variable] - d_min[variable]).clamp_min(1e-8)
        global_mean[variable] = ((values - d_min[variable]) / scale).mean()
    return d_min, d_max, global_mean

def create_collate_fn_mimic_v31(args, d_min, d_max):
    """Dataset experiment entry point."""
    def collate_fn(batch):
        B, M, N = len(batch), args.npatch, args.ndim
        data_range = d_max - d_min
        data_range[data_range == 0] = 1.0
        max_obs_len = max_pred_len = 0
        L_limit = args.max_points
        for (_, tt, _, mask) in batch:
            for p in range(M):
                p_len = ((tt >= p*args.stride) & (tt < p*args.stride+args.patch_size) & (tt < args.history)).sum().item()
                max_obs_len = max(max_obs_len, min(p_len, L_limit))
            prediction = (tt >= args.history) & (tt < args.history + args.pred_window)
            max_pred_len = max(max_pred_len, prediction.sum().item())
        max_obs_len = max(max_obs_len, 1)
        max_pred_len = max(max_pred_len, 1)

        X_raw = torch.zeros(B, M, max_obs_len, N)
        truth_ts = torch.zeros(B, M, max_obs_len, 1)
        m_tensor = torch.zeros(B, M, max_obs_len, N)
        targets = torch.zeros(B, max_pred_len, N)
        t_masks = torch.zeros(B, max_pred_len, N)
        p_ts = torch.zeros(B, max_pred_len)

        for i, (rid, tt, vals, m) in enumerate(batch):
            vals_n = (vals - d_min) / data_range
            vals_n = torch.nan_to_num(vals_n, nan=0.0)
            for p in range(M):
                p_idx = torch.where((tt >= p*args.stride) & (tt < p*args.stride+args.patch_size) & (tt < args.history))[0][-L_limit:]
                L = len(p_idx)
                if L > 0:
                    X_raw[i, p, :L, :] = vals_n[p_idx].float()
                    truth_ts[i, p, :L, 0] = (tt[p_idx]/args.time_max).float()
                    m_tensor[i, p, :L, :] = m[p_idx].float()
            pred_idx = torch.where(
                (tt >= args.history) & (tt < args.history + args.pred_window)
            )[0]
            L_p = len(pred_idx)
            if L_p > 0:
                targets[i, :L_p, :] = vals_n[pred_idx].float()
                t_masks[i, :L_p, :] = m[pred_idx].float()
                p_ts[i, :L_p] = (tt[pred_idx]/args.time_max).float()

        return {'X_raw': X_raw, 'truth_time_steps': truth_ts,
                'time_steps_to_predict': p_ts, 'mask': m_tensor,
                'targets': targets, 'target_masks': t_masks}
    return collate_fn



def get_sample_losses(pred, targets, target_masks):
    sq_error = ((targets - pred)**2) * target_masks
    valid = target_masks.sum(dim=(1, 2)) + 1e-8
    return sq_error.sum(dim=(1, 2)) / valid

def train_epoch(model, calib, dataloader, optimizer, device, global_mean):
    model.train()
    calib.train()
    total_loss = 0
    for batch in dataloader:
        X_raw = batch['X_raw'].to(device)
        mask = batch['mask'].to(device)
        ts = batch['time_steps_to_predict'].to(device)
        truth_ts = batch['truth_time_steps'].to(device)
        targets = batch['targets'].to(device)
        target_masks = batch['target_masks'].to(device)

        X_cons, X_agg, tm_cons, tm_agg, taus, rhos, patch_scores, keep_cons, keep_agg = apply_addp_routing_calibrated(
            X_raw, mask, global_mean, calib)

        X_concat = torch.cat([X_cons, X_agg], dim=0)
        tm_concat = torch.cat([tm_cons, tm_agg], dim=0)
        ts_concat = torch.cat([ts, ts], dim=0)
        truth_ts_concat = torch.cat([truth_ts, truth_ts], dim=0)
        mask_concat = torch.cat([mask, mask], dim=0)

        out_concat = model.forecasting(ts_concat, X_concat, truth_ts_concat, mask_concat, tm_concat).squeeze(0)
        B = targets.shape[0]
        out_cons, out_agg = out_concat[:B], out_concat[B:]

        loss_cons = get_sample_losses(out_cons, targets, target_masks)
        loss_agg = get_sample_losses(out_agg, targets, target_masks)
        stacked_losses = torch.stack([loss_cons, loss_agg], dim=1)
        taus_b = taus.unsqueeze(1)
        max_loss = stacked_losses.max(dim=1, keepdim=True)[0]
        dro_loss = taus_b * torch.log(torch.mean(torch.exp((stacked_losses - max_loss) / taus_b), dim=1, keepdim=True) + 1e-8) + max_loss
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
    return total_loss / len(dataloader)

def calculate_metric_totals(pred, targets, masks):
    sq_error = ((pred - targets) ** 2) * masks
    abs_error = torch.abs(pred - targets) * masks
    return sq_error.sum().item(), abs_error.sum().item(), masks.sum().item()

def test_epoch(model, dataloader, device):
    model.eval()
    total_sq_error = 0.0
    total_abs_error = 0.0
    total_observations = 0.0
    with torch.no_grad():
        for batch in dataloader:
            X_raw = batch['X_raw'].to(device)
            ts = batch['time_steps_to_predict'].to(device)
            truth_ts = batch['truth_time_steps'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)
            t_mask = batch['target_masks'].to(device)
            outputs = model.forecasting(ts, X_raw, truth_ts, mask, topo_pruning_mask=None)
            sq_error, abs_error, observations = calculate_metric_totals(
                outputs.squeeze(0), targets, t_mask
            )
            total_sq_error += sq_error
            total_abs_error += abs_error
            total_observations += observations
    denominator = max(total_observations, 1.0)
    mse = total_sq_error / denominator
    mae = total_abs_error / denominator
    return mse, mae, math.sqrt(mse)



def main():
    print("=" * 70)
    print("ADDP MIMIC-III - Learnable Calibrators")
    print("ADDP with constrained learnable calibration")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = Args(device)
    SEED = int(os.environ.get('ADDP_SEED', '0'))
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    project_root = os.environ.get('ADDP_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))
    candidates = [
        os.path.join(project_root, 'data', 'mimic', 'processed', 'mimic.pt'),
        os.path.join(project_root, 'data', 'mimic', 'mimic_fixed_float32.pt'),
    ]
    data_path = next((path for path in candidates if os.path.exists(path)), None)
    if data_path is None:
        raise FileNotFoundError(
            "MIMIC-III data not found. See data/mimic/README.md and place the "
            "credentialed file at data/mimic/processed/mimic.pt."
        )
    full_data = torch.load(data_path, map_location='cpu')

    seen_data, test_data = train_test_split(full_data, train_size=0.8, random_state=42, shuffle=True)
    train_data, val_data = train_test_split(seen_data, train_size=0.75, random_state=42, shuffle=False)
    print(f"Split: train={len(train_data)}, validation={len(val_data)}, test={len(test_data)}")

    d_min, d_max, global_mean = get_data_min_max_and_mean(train_data, ndim=args.ndim)
    collate_v31 = create_collate_fn_mimic_v31(args, d_min, d_max)
    train_loader = DataLoader(MimicDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate_v31)
    val_loader = DataLoader(MimicDataset(val_data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_v31)
    test_loader = DataLoader(MimicDataset(test_data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_v31)

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
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_mse = float('inf')
    best_model_state = None
    best_calib_state = None
    best_epoch = 0
    patience_cnt = 0
    patience_limit = 10
    save_path = f'mimic_addp_seed{SEED}_best.pth'

    print(f"\nTraining for at most {args.epoch} epochs (patience={patience_limit})")
    for epoch in range(args.epoch):
        train_loss = train_epoch(model, calib, train_loader, optimizer, device, global_mean)
        val_mse, val_mae, val_rmse = test_epoch(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_mse)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_calib_state = copy.deepcopy(calib.state_dict())
            torch.save(
                {'model': best_model_state, 'calibration': best_calib_state,
                 'seed': SEED, 'best_epoch': best_epoch},
                save_path,
            )
            indicator = " (best)"
            patience_cnt = 0
        else:
            indicator = ""
            patience_cnt += 1
        print(f"Epoch {epoch:2d}: DRO Loss = {train_loss:.5f}, Val MSE = {val_mse:.5f} (LR={current_lr:.6f}) {indicator}")
        if epoch >= 25 and patience_cnt >= patience_limit:
            print("Early stopping")
            break

    print(f"\nBest epoch: {best_epoch}; validation MSE: {best_val_mse:.6f}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        calib.load_state_dict(best_calib_state)

    t_mse, t_mae, t_rmse = test_epoch(model, test_loader, device)

    print("\n" + "=" * 70)
    print("\nTest results")
    print(f"  Test MSE  : {t_mse:.5f}")
    print(f"  Test MAE  : {t_mae:.5f}")
    print(f"  Test RMSE : {t_rmse:.5f}")
    print("=" * 70)

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
    print(f"  b_c init={desc_init['b_c']:.4f} -> final={desc_final['b_c']:.4f}")
    print(f"  b_temperature init={desc_init['b_tau']:.4f} -> final={desc_final['b_tau']:.4f}")

if __name__ == "__main__":
    main()
