import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
from torch import nn
import time
from models.gat_gcn_SA_LSTM_GSA_transformer_ge_mut_cross import \
    GAT_GCN_SA_LSTM_GSA_Transformer_ge_mut_cross

from utils_smiles import *
import datetime
import argparse
from utils_smiles import r2
import random


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(1000)
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval, model_st):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    loss_ae = nn.MSELoss()
    avg_loss = []
    weight_fn = 0.01
    weight_ae = 2
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader, model_st):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            output, _ = model(data)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name):

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    path = './result_SA_GCA_GCN_SA2GSA_cross_GDSC/'
    # path = './result/'
    if not os.path.exists(path):
        os.makedirs(path)
    for model in modeling:
        model_st = model.__name__
        dataset = 'GDSC'
        train_losses = []
        val_losses = []
        val_pearsons = []
        print('\nrunning on ', model_st + '_' + dataset )
        processed_data_file_train = './data/processed/' + dataset + '_train_mix' + '.pt'
        # print(processed_data_file_train)
        processed_data_file_val = './data/processed/' + dataset + '_val_mix' + '.pt'
        processed_data_file_test = './data/processed/' + dataset + '_test_mix' + '.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (
        not os.path.isfile(processed_data_file_test))):
            print('please run create_data.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root='./data', dataset=dataset + '_train_mix')
            val_data = TestbedDataset(root='./data', dataset=dataset + '_val_mix')
            test_data = TestbedDataset(root='./data', dataset=dataset + '_test_mix')

            # make data PyTorch mini-batch processing ready
            train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
            print("CPU/GPU: ", torch.cuda.is_available())
                    
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            print(device)
            model = model().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            best_mse = 1000
            best_pearson = 1
            best_epoch = -1
            model_file_name = path + 'model_' + model_st + '_' + dataset + '.model'
            result_file_name = path + 'result_' + model_st + '_' + dataset + '.csv'
            loss_fig_name = path + 'model_' + model_st + '_' + dataset + '_loss'
            pearson_fig_name = path + 'model_' + model_st + '_' + dataset + '_pearson'
            start_time = time.time()
            for epoch in range(num_epoch):
                train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval, model_st)
                G,P = predicting(model, device, val_loader, model_st)
                ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
                            
                G_test,P_test = predicting(model, device, test_loader, model_st)
                ret_test = [rmse(G_test, P_test), mse(G_test, P_test), pearson(G_test, P_test),
                            spearman(G_test, P_test),
                            r2(G_test, P_test)]
                print(pearson(G_test, P_test))
                train_losses.append(train_loss)
                val_losses.append(ret[1])
                val_pearsons.append(ret[2])


                if ret_test[1] < best_mse:
                    torch.save(model.state_dict(), model_file_name)
                    with open(result_file_name, 'w') as f:
                        f.write(','.join(map(str, ret_test)))
                    best_epoch = epoch + 1
                    best_rmse = ret_test[0]
                    best_mse = ret_test[1]
                    best_pearson = ret_test[2]
                    best_spearman = ret_test[3]
                    best_r2 = ret_test[4]
                    print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse, model_st, dataset)
                else:
                    print(' no improvement since epoch ', best_epoch,
                          '; best_mse, best_rmse,best pearson ,best_spearman, best_r2:', best_mse, best_rmse,
                          best_pearson,
                          best_spearman, best_r2, model_st, dataset)
            end_time = time.time()
            # 计算程序的运行时间
            run_time = end_time - start_time
            # 输出运行时间
            print("程序运行时间为：", run_time, "秒")
            draw_loss(train_losses, val_losses, loss_fig_name)
            draw_pearson(val_pearsons, pearson_fig_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0,     help='0: Transformer_ge_mut_meth, 1: Transformer_ge_mut, 2: Transformer_meth_mut, 3: Transformer_meth_ge, 4: Transformer_ge, 5: Transformer_mut, 6: Transformer_meth')
    parser.add_argument('--train_batch', type=int, required=False, default=32,  help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=32, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=32, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=200, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')

    args = parser.parse_args()

    modeling = [GAT_GCN_SA_LSTM_GSA_Transformer_ge_mut_cross][args.model]
    model = [modeling]
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name

    main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name)