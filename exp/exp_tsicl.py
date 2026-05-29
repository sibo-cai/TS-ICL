from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings('ignore')


class Exp_tsicl(Exp_Basic):
    def __init__(self, args):
        super(Exp_tsicl, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_loader_list = data_provider(self.args, flag)
        return data_loader_list

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _normalize_xy(self, batch_x, batch_y, global_scaler=None):
        # [batch(*stop), day, time]
        means = batch_x.mean(dim=2, keepdim=True)
        batch_x = batch_x - means
        batch_x[torch.abs(batch_x) < 1e-6] = 0
        stdev = torch.sqrt(
            torch.var(batch_x, dim=2, keepdim=True, unbiased=False)
        )
        stdev = torch.clamp(stdev, min=1e-6)
        batch_x /= stdev    # [batch(*stop), day, 1]

        means = batch_y.mean(dim=2, keepdim=True)
        batch_y = batch_y - means
        batch_y[torch.abs(batch_y) < 1e-6] = 0
        stdev = torch.sqrt(
            torch.var(batch_y, dim=2, keepdim=True, unbiased=False)
        )
        stdev = torch.clamp(stdev, min=1e-6)
        batch_y /= stdev    # [batch(*stop), day, 1]

        return batch_x, batch_y

        if global_scaler is None:
            batch_y = batch_y - means
            batch_y[torch.abs(batch_y) < 1e-6] = 0
            batch_y /= stdev
        else:
            mask = stdev < 1e-5
            normal_batch_y = batch_y - means
            normal_batch_y[torch.abs(normal_batch_y) < 1e-6] = 0
            normal_batch_y /= stdev

            # average
            # g_mean = torch.full_like(means, fill_value=np.mean(global_scaler.mean_).item())
            # g_std = torch.full_like(stdev, fill_value=np.mean(global_scaler.scale_).item())
            # first stop
            g_mean = torch.full_like(means, fill_value=global_scaler.mean_[0])
            g_std = torch.full_like(stdev, fill_value=global_scaler.scale_[0])
            g_batch_y = batch_y - g_mean
            g_batch_y[torch.abs(g_batch_y) < 1e-6] = 0
            g_batch_y /= g_std

            batch_y = torch.where(mask, g_batch_y, normal_batch_y)

        return batch_x, batch_y


    def vali(self, vali_loader_list, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for vali_loader_i, vali_loader_items in enumerate(vali_loader_list):
                vali_loader = vali_loader_items[0]
                n_samples = vali_loader_items[1]
                single_eval_pos = vali_loader_items[2]
                # print(f'Vali loader {vali_loader_i}, n_samples={n_samples}, single_eval_pos={single_eval_pos}')
                # for i, (batch_x, batch_y) in enumerate(tqdm(vali_loader, desc="validation")):
                for i, (batch_x, batch_y) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x, batch_y = self._normalize_xy(batch_x, batch_y)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, single_eval_pos:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :single_eval_pos, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, dec_inp, single_eval_pos)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp, single_eval_pos)[0]
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, single_eval_pos:, :]

                    pred = outputs.detach()
                    true = batch_y.detach()

                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_loader_list = self._get_data(flag='train')
        vali_loader_list = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        steps_per_epoch = sum([len(x[0]) for x in train_loader_list])
        model_scheduler = None
        if self.args.lradj == "step":
            model_scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate,
            )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_train_loss = []
        epoch_valid_loss = []
        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr']))
            for train_loader_i, train_loader_items in enumerate(tqdm(train_loader_list, desc='training')):
                train_loader = train_loader_items[0]
                n_samples = train_loader_items[1]
                single_eval_pos = train_loader_items[2]
                # print(f'Train loader {train_loader_i}, n_samples {n_samples}, single_eval_pos {single_eval_pos}')
                # for i, (batch_x, batch_y) in enumerate(tqdm(train_loader, desc='training')):
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device, non_blocking=True)
                    batch_y = batch_y.float().to(self.device, non_blocking=True)

                    # normalization
                    batch_x, batch_y = self._normalize_xy(batch_x, batch_y)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, single_eval_pos:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :single_eval_pos, :], dec_inp], dim=1).float()

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, dec_inp, single_eval_pos)[0]
                            batch_y = batch_y[:, single_eval_pos:, :]
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        outputs = self.model(batch_x, dec_inp, single_eval_pos)[0]
                        batch_y = batch_y[:, single_eval_pos:, :]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                    # adjust lr by step
                    if self.args.lradj == "step":
                        model_scheduler.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader_list, criterion)

            epoch_train_loss.append(train_loss.item())
            epoch_valid_loss.append(vali_loss.item())

            print("Epoch: {0}, Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(epoch + 1, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust lr by epoch
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        print("Training time: {}".format(time.time() - time_now))
        print(f"Epoch train loss: {epoch_train_loss}")
        print(f"Epoch valid loss: {epoch_valid_loss}")
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_loader_list = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for test_loader_i, test_loader_items in enumerate(test_loader_list):
                test_loader = test_loader_items[0]
                n_samples = test_loader_items[1]
                single_eval_pos = test_loader_items[2]
                # print(f'Test loader {test_loader_i}, n_samples={n_samples}, single_eval_pos={single_eval_pos}')
                # for i, (batch_x, batch_y) in enumerate(tqdm(test_loader, desc='testing')):
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x, batch_y = self._normalize_xy(batch_x, batch_y)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, single_eval_pos:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :single_eval_pos, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, dec_inp, single_eval_pos)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp, single_eval_pos)[0]

                    batch_y = batch_y[:, single_eval_pos:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                    # if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, -1, :], true[0, -1, :]), axis=0)
                    pd = np.concatenate((input[0, -1, :], pred[0, -1, :]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

    def test_ygtrip(self, setting):
        test_loader_list = self._get_data(flag='test')
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_ygtrip_results/' + setting + self.args.ygtrip_day_mode + self.args.ygtrip_data_id + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for test_loader_i, test_loader_items in enumerate(test_loader_list):
                test_loader = test_loader_items[0]
                # n_samples = test_loader_items[1]
                # single_eval_pos = test_loader_items[2]
                # print(f'Test loader {test_loader_i}, n_samples={n_samples}, single_eval_pos={single_eval_pos}')
                # for i, (batch_x, batch_y) in enumerate(tqdm(test_loader, desc='testing')):
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    # [batch, stop, day, time]
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_num, n_stops, n_days, _ = batch_x.shape
                    # [batch * stop, day, time]
                    batch_x = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1])
                    batch_y = batch_y.reshape(-1, batch_y.shape[-2], batch_y.shape[-1])
                    single_eval_pos = n_days - 1

                    # normalization
                    batch_x, batch_y = self._normalize_xy(batch_x, batch_y, test_loader.dataset.get_scaler())

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, single_eval_pos:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :single_eval_pos, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, dec_inp, single_eval_pos)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp, single_eval_pos)[0]

                    # start from single eval pos
                    batch_y = batch_y[:, single_eval_pos:, :].to(self.device)

                    # [batch, stop, time]
                    outputs = outputs.reshape(batch_num, n_stops, outputs.shape[-2], outputs.shape[-1]).squeeze(2)
                    batch_y = batch_y.reshape(batch_num, n_stops, batch_y.shape[-2], batch_y.shape[-1]).squeeze(2)

                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                    # if i % 20 == 0:
                    # [batch, stop, day, time] -> [batch, stop, time]
                    # input = batch_x.reshape(batch_num, n_stops, batch_x.shape[-2], batch_x.shape[-1])[:,:,-1,:]
                    # input = input.detach().cpu().numpy()
                    # gt = np.concatenate((input[0, 0, :], true[0, 0, :]), axis=0)
                    # pd = np.concatenate((input[0, 0, :], pred[0, 0, :]), axis=0)
                    gt = true[0, 0, :]
                    pd = pred[0, 0, :]
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './ygtrip_results/' + setting + self.args.ygtrip_day_mode + self.args.ygtrip_data_id + '/'
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

    def custom_test(self, setting, csv_file, n_samples, single_eval_pos, n_estimators=8, shuffle=True, use_norm=True, retrieval_ensemble=False):
        print('loading model tspfn')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
        print(
            "number of model params",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        print('data file', csv_file)

        self.model.eval()
        with torch.no_grad():
            test_data = pd.read_csv(csv_file).values
            test_data = np.transpose(test_data)
            print('test shape', test_data.shape, 'n_samples', n_samples, 'single_eval_pos', single_eval_pos)
            test_data = np.expand_dims(test_data, axis=0)
            test_data = torch.from_numpy(test_data).float().to(self.device)

            print('n_estimators', n_estimators, 'shuffle', shuffle, 'normalization', use_norm)

            outputs = []
            attns = []
            outputs_2nd = []
            attns_2nd = []
            shuffle_test_data = test_data
            for i_estimator in range(n_estimators):
                if shuffle:
                    train_idx = torch.randperm(single_eval_pos)
                    test_idx = torch.tensor([x for x in range(single_eval_pos, n_samples)])
                    idx = torch.cat([train_idx, test_idx], dim=0)
                    print('shuffle', idx)
                    idx = idx.to(self.device)
                    shuffle_test_data = torch.index_select(test_data, dim=1, index=idx)

                # split
                batch_x = shuffle_test_data[:, :, :self.args.seq_len]
                batch_y = shuffle_test_data[:, :, self.args.seq_len-self.args.label_len:]

                # norm
                use_norm_dim = 2
                if use_norm:
                    means = batch_x.mean(dim=use_norm_dim, keepdim=True)
                    batch_x = batch_x - means
                    stdev = torch.sqrt(torch.var(batch_x, dim=use_norm_dim, keepdim=True, unbiased=False) + 1e-20)
                    batch_x /= stdev

                t = batch_y[:, :single_eval_pos, :]
                if use_norm:
                    means = t.mean(dim=use_norm_dim, keepdim=True)
                    t = t - means
                    stdev = torch.sqrt(torch.var(t, dim=use_norm_dim, keepdim=True, unbiased=False) + 1e-20)
                    t /= stdev

                dec_inp = torch.zeros_like(batch_y[:, single_eval_pos:, :]).float()
                dec_inp = torch.cat([t, dec_inp], dim=1).float().to(self.device)

                start_time = time.time()
                output, attn = self.model(batch_x, dec_inp, single_eval_pos)
                end_time = time.time()
                run_time = end_time - start_time
                print(i_estimator, "custom test runtime：%f sec" % run_time)

                if retrieval_ensemble:
                    # retrival-based forward
                    # attn: [n_layers, batch, n_headd, n_test, n_train]
                    # last layer, batch=1
                    attn_last_layer = attn[-1, 0]   # [n_heads, n_test, n_train]
                    # mean twice
                    # attention head mean
                    attn_last_layer = torch.mean(attn_last_layer, dim=0)    # [n_test, n_train]
                    # query mean
                    attn_last_layer = torch.mean(attn_last_layer, dim=0)    # [n_train]

                    # attn_thres = 0.019
                    # selected_idx = torch.nonzero(attn_last_layer >= attn_thres).squeeze()
                    top_k = int(single_eval_pos * 0.1)
                    selected_idx = torch.topk(attn_last_layer, top_k)[1]
                    print('retrieve samples using attention', selected_idx.shape[0])
                    print(selected_idx)
                    test_idx = torch.tensor([x for x in range(single_eval_pos, n_samples)])
                    idx = torch.cat([selected_idx, test_idx], dim=0)

                    batch_x_2nd = batch_x[:, idx, :]
                    dec_inp_2nd = dec_inp[:, idx, :]
                    output_2nd, attn_2nd = self.model(batch_x_2nd, dec_inp_2nd, selected_idx.shape[0])

                # denorm
                if use_norm:
                    # output = output * (
                    #     (stdev.mean(dim=1, keepdim=True)*1.0).repeat(1, 1, self.args.pred_len + self.args.label_len))
                    # output = output + (
                    #     (means.mean(dim=1, keepdim=True)+0.0).repeat(1, 1, self.args.pred_len + self.args.label_len))
                    output = output * (
                        (stdev[:,-1:,:]).repeat(1, 1, self.args.pred_len + self.args.label_len))
                    output = output + (
                        (means[:,-1:,:]).repeat(1, 1, self.args.pred_len + self.args.label_len))

                    if retrieval_ensemble:
                        output_2nd = output_2nd * (
                            (stdev.mean(dim=1, keepdim=True)*1.0).repeat(1, 1, self.args.pred_len + self.args.label_len))
                        output_2nd = output_2nd + (
                            (means.mean(dim=1, keepdim=True)+0.0).repeat(1, 1, self.args.pred_len + self.args.label_len))

                outputs.append(output)
                attns.append(attn)
                if retrieval_ensemble:
                    outputs_2nd.append(output_2nd)
                    attns_2nd.append(attn_2nd)

            batch_y = batch_y[:, single_eval_pos:, :]
            true = batch_y.repeat(n_estimators, 1, 1, ).detach().cpu().numpy()

            outputs = torch.cat(outputs, axis=0)
            outputs_mean = torch.mean(outputs, dim=0).detach().cpu().numpy()
            attns = torch.stack(attns)
            pred = outputs.detach().cpu().numpy()
            mae, mse, rmse, mape, mspe = metric(pred, true)
            print('1st mse:{}, mae:{}'.format(mse, mae))

            if retrieval_ensemble:
                outputs_2nd = torch.cat(outputs_2nd, axis=0)
                outputs_2nd_mean = torch.mean(outputs_2nd, dim=0).detach().cpu().numpy()
                attns_2nd = torch.stack(attns_2nd)
                pred_2nd = outputs_2nd.detach().cpu().numpy()
                mae_2nd, mse_2nd, rmse_2nd, mape_2nd, mspe_2nd  = metric(pred_2nd, true)
                print('2nd mse:{}, mae:{}'.format(mse_2nd, mae_2nd))

            for eval_sample_i in range(n_samples - single_eval_pos):
                fig, ax = plt.subplots()
                # for i_estimator in range(n_estimators):
                #     ax.plot(pred[i_estimator, eval_sample_i], label=f"Predictions-{i_estimator}", linewidth=1)
                ax.plot(true[i_estimator, eval_sample_i], label="Ground Truth")
                ax.plot(outputs_mean[eval_sample_i], label="Predictions-mean")
                if retrieval_ensemble:
                    ax.plot(outputs_2nd_mean[eval_sample_i], label="Predictions-mean_2nd")
                ax.legend()
                fig.text(0.2, 0.025, f'1st mae:{mae}, mse:{mse}')
                if retrieval_ensemble:
                    fig.text(0.2, 0.0, f'2nd mae:{mae_2nd}, mse:{mse_2nd}')
                plt.savefig(f'{csv_file}_eval_{eval_sample_i}.pdf', bbox_inches='tight')

            # save attns
            print('attns shape:', attns.shape)
            torch.save(attns, f'{csv_file}_attns.pth')
            if retrieval_ensemble:
                print('attns_2nd shape:', attns_2nd.shape)
                torch.save(attns_2nd, f'{csv_file}_attns_2nd.pth')
