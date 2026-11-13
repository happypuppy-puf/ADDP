import os
from pathlib import Path
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
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
        self.hid_dim = 32
        self.ndim = 5
        self.nlayer = 1
        self.te_dim = 10
        self.node_dim = 10
        self.nhead = 2
        self.tf_layer = 1
        self.hop = 1
        self.outlayer = "Linear"
        self.n_months = 48
        self.history = 24.0
        self.pred_window = 1.0
        self.patch_size = 2.0
        self.stride = 2.0
        self.time_max = 48.0
        self.npatch = int(np.ceil((self.history - self.patch_size) / self.stride)) + 1


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
        attn = F.softmax(Filter_mask, dim=-2) * mask_X
        attn = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)
        Filter_seqnorm = attn.view(N_total, Lx, self.ttcn_dim, -1)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        return torch.relu(torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) + self.T_bias)
    def IMTS_Model(self, x, mask_X, topo_pruning_mask=None):
        N_vars, M_patch = self.N, self.M
        B = x.shape[0] // (N_vars * M_patch)
        transformer_mask = (mask_X.sum(dim=1) == 0).view(B * N_vars, M_patch)
        if topo_pruning_mask is not None:
            topo_ignore = (1 - topo_pruning_mask).bool().repeat_interleave(N_vars, dim=0)
            transformer_mask = transformer_mask | topo_ignore
        all_empty_seqs = transformer_mask.all(dim=1)
        transformer_mask[all_empty_seqs, 0] = False
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
            nodevec2 = nodevec2 + (x_gate2 * self.nodevec_linear2[layer](x)).permute(0,2,3,1)
            valid_nodes = (~true_empty_mask).float()
            gnn_mask = valid_nodes.unsqueeze(-1) * valid_nodes.unsqueeze(-2)
            raw_adp = F.relu(torch.matmul(nodevec1, nodevec2))
            raw_adp = raw_adp.masked_fill(gnn_mask == 0, -1e9)
            adp = F.softmax(raw_adp, dim=-1)
            adp = torch.nan_to_num(adp, nan=0.0)
            new_supports = self.supports + [adp]
            x = self.gconv[layer](x.permute(0,3,1,2), new_supports).permute(0, 2, 3, 1)
            if layer > 0: x = x_last + x
        return self.temporal_agg(x.reshape(B, N_vars, -1))
    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask, topo_pruning_mask=None):
        B, M, L_in, N = X.shape
        X = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        if truth_time_steps.shape[-1] == 1:
            truth_time_steps = truth_time_steps.expand(-1, -1, -1, N)
            truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        elif truth_time_steps.shape[-1] == 3:
            truth_time_steps = truth_time_steps.unsqueeze(-2).expand(-1, -1, -1, N, -1)
            truth_time_steps = truth_time_steps.permute(0, 3, 1, 2, 4).reshape(-1, L_in, 3)
        mask = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        X = torch.cat([X, self.LearnableTE(truth_time_steps)], dim=-1)
        h = self.IMTS_Model(X, mask, topo_pruning_mask)
        L_pred = time_steps_to_predict.shape[1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
        decoder_input = torch.cat([h, self.LearnableTE(time_steps_to_predict)], dim=-1)
        return self.decoder(decoder_input).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)




def load_ushcn_data(file_path):
    df = pd.read_csv(file_path)
    stations = df['ID'].unique()
    return df, stations

def preprocess_ushcn_data(df):
    var_cols = [f'Value_{i}' for i in range(5)]
    mask_cols = [f'Mask_{i}' for i in range(5)]
    stations_data = {}
    for station_id in df['ID'].unique():
        station_df = df[df['ID'] == station_id].copy()
        station_df = station_df.sort_values(['Time'])
        values = station_df[var_cols].values.astype(np.float32)
        masks = station_df[mask_cols].values.astype(np.float32)
        timestamps = station_df['Time'].values.astype(np.float32) * (48.0 / 200.0)
        values[masks == 0] = 0
        stations_data[station_id] = {'values': values, 'masks': masks, 'timestamps': timestamps}
    return stations_data

def get_global_statistics(train_data_dict, ndim=5):
    all_vals = [[] for _ in range(ndim)]
    for st, data in train_data_dict.items():
        v, m = data['values'], data['masks']
        for d in range(ndim):
            v_d = v[:, d][m[:, d] == 1]
            if len(v_d) > 0: all_vals[d].extend(v_d)
    mu_global = torch.zeros(ndim)
    for d in range(ndim):
        if len(all_vals[d]) > 0: mu_global[d] = float(np.mean(all_vals[d]))
    return mu_global

class USHCNDataset(Dataset):
    def __init__(self, data_dict, args):
        self.args = args
        self.samples = []
        for station_id, station_data in data_dict.items():
            values, masks, timestamps = station_data['values'], station_data['masks'], station_data['timestamps']
            for st in range(0, int(args.n_months - args.history - args.pred_window + 1), int(args.pred_window)):
                hist_mask = (timestamps >= st) & (timestamps < st + args.history)
                pred_mask = (timestamps >= st + args.history) & (timestamps < st + args.history + args.pred_window)
                if np.sum(masks[pred_mask]) > 0 and np.sum(masks[hist_mask]) > 0:
                    self.samples.append({
                        'hist_times': timestamps[hist_mask] / args.time_max,
                        'hist_vals': values[hist_mask],
                        'hist_masks': masks[hist_mask],
                        'pred_times': timestamps[pred_mask] / args.time_max,
                        'pred_vals': values[pred_mask],
                        'pred_masks': masks[pred_mask],
                        'st': st / args.time_max,
                    })
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def create_collate_fn_v31(args, global_mean):
    def collate_fn(batch):
        B = len(batch)
        M = args.npatch
        N = args.ndim

        max_obs_len = max([((s['hist_times']*args.time_max - s['st']*args.time_max) < args.history).sum() for s in batch])
        max_pred_len = max([len(s['pred_times']) for s in batch])
        max_obs_len, max_pred_len = max(max_obs_len, 1), max(max_pred_len, 1)

        X_raw = torch.zeros(B, M, max_obs_len, N)
        truth_time_steps = torch.zeros(B, M, max_obs_len, 1)
        mask = torch.zeros(B, M, max_obs_len, N)
        targets = torch.zeros(B, max_pred_len, N)
        target_masks = torch.zeros(B, max_pred_len, N)
        time_steps_to_predict = torch.zeros(B, max_pred_len)

        for i, s in enumerate(batch):
            abs_tt = s['hist_times']
            rel_tt = abs_tt * args.time_max - (s['st'] * args.time_max)
            for p in range(M):
                start = p * args.stride
                end = start + args.patch_size
                p_idx = np.where((rel_tt >= start) & (rel_tt < end))[0]
                L = len(p_idx)
                if L > 0:
                    X_raw[i, p, :L, :] = torch.FloatTensor(s['hist_vals'][p_idx])
                    truth_time_steps[i, p, :L, 0] = torch.FloatTensor(abs_tt[p_idx])
                    mask[i, p, :L, :] = torch.FloatTensor(s['hist_masks'][p_idx])
            L_p = len(s['pred_times'])
            if L_p > 0:
                targets[i, :L_p, :] = torch.FloatTensor(s['pred_vals'])
                target_masks[i, :L_p, :] = torch.FloatTensor(s['pred_masks'])
                time_steps_to_predict[i, :L_p] = torch.FloatTensor(s['pred_times'])

        return {'X_raw': X_raw, 'truth_time_steps': truth_time_steps,
                'mask': mask, 'targets': targets, 'target_masks': target_masks,
                'time_steps_to_predict': time_steps_to_predict}
    return collate_fn



def get_sample_losses(pred, targets, target_masks):
    sq_error = ((targets - pred)**2) * target_masks
    valid = target_masks.sum(dim=(1, 2)) + 1e-8
    return sq_error.sum(dim=(1, 2)) / valid

def train_epoch(model, calib, dataloader, optimizer, device):
    model.train()
    calib.train()
    total_loss = 0
    for batch in dataloader:
        X_raw = batch['X_raw'].to(device)
        mask = batch['mask'].to(device)
        time_steps = batch['time_steps_to_predict'].to(device)
        truth_time_steps = batch['truth_time_steps'].to(device)
        targets = batch['targets'].to(device)
        target_masks = batch['target_masks'].to(device)

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
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_mse_sum = torch.zeros(5).to(device)
    total_mask_count = torch.zeros(5).to(device)
    with torch.no_grad():
        for batch in dataloader:
            X_raw = batch['X_raw'].to(device)
            ts = batch['time_steps_to_predict'].to(device)
            truth_ts = batch['truth_time_steps'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)
            target_mask = batch['target_masks'].to(device)
            outputs = model.forecasting(ts, X_raw, truth_ts, mask, topo_pruning_mask=None)
            pred = outputs.squeeze(0)
            N_dim = targets.shape[-1]
            sq_error = ((pred - targets) ** 2) * target_mask
            total_mse_sum += sq_error.reshape(-1, N_dim).sum(dim=0)
            total_mask_count += target_mask.reshape(-1, N_dim).sum(dim=0)
    error_var_avg = total_mse_sum / (total_mask_count + 1e-8)
    n_avai_var = torch.count_nonzero(total_mask_count)
    return (error_var_avg.sum() / n_avai_var).item()



global_mean_for_calib = None

def main():
    print("=" * 70)
    print("ADDP - Learnable Calibrators - USHCN")
    print("ADDP with constrained learnable calibration")
    print("=" * 70)

    global global_mean_for_calib

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = Args(device)
    SEED = int(os.environ.get('ADDP_SEED', '0'))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    project_root = os.environ.get('ADDP_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))
    data_path = os.path.join(project_root, 'data', 'ushcn', 'small_chunked_sporadic.csv')
    df, all_stations = load_ushcn_data(data_path)
    print(f"Dimensions: {args.ndim}, patches: {args.npatch}")

    seen_stations, test_stations = train_test_split(all_stations, train_size=0.8, random_state=42, shuffle=True)
    train_stations, val_stations = train_test_split(seen_stations, train_size=0.75, random_state=42, shuffle=False)
    print(
        f"Stations: train={len(train_stations)}, "
        f"validation={len(val_stations)}, test={len(test_stations)}"
    )

    stations_data = preprocess_ushcn_data(df)
    train_data = {s: stations_data[s] for s in train_stations}
    val_data = {s: stations_data[s] for s in val_stations}
    test_data = {s: stations_data[s] for s in test_stations}
    mu_global = get_global_statistics(train_data, args.ndim)
    global_mean_for_calib = mu_global.to(device)

    batch_size = 64
    collate_v31 = create_collate_fn_v31(args, mu_global)
    train_loader = DataLoader(USHCNDataset(train_data, args), batch_size=batch_size, shuffle=True, collate_fn=collate_v31)
    val_loader = DataLoader(USHCNDataset(val_data, args), batch_size=batch_size, shuffle=False, collate_fn=collate_v31)
    test_loader = DataLoader(USHCNDataset(test_data, args), batch_size=batch_size, shuffle=False, collate_fn=collate_v31)

    model = ADDPModel(args, use_future_timestamps=True).to(device)
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
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_model_state = None
    best_calib_state = None
    best_epoch = 0
    epochs, patience, no_improve_cnt = 50, 10, 0
    save_path = f'ushcn_addp_seed{SEED}_best.pth'

    print(f"\nTraining for at most {epochs} epochs (patience={patience})")
    for epoch in range(epochs):
        train_loss = train_epoch(model, calib, train_loader, optimizer, device)
        val_mse = evaluate_epoch(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}: DRO Loss = {train_loss:.5f}, Val MSE = {val_mse:.5f} (LR={current_lr:.6f})")

        scheduler.step(val_mse)
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

    test_mse = evaluate_epoch(model, test_loader, device)

    print("\n" + "=" * 60)
    print("\nTest results")
    print(f"  Test MSE : {test_mse:.6f}")
    print(f"  Test RMSE: {np.sqrt(test_mse):.6f}")
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
