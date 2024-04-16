import os
import sys
import gc
import dgl
import time
import pickle
import torch
import random
import timeit
import warnings
import argparse
import numpy as np
from tqdm import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable 
from torch_geometric.data import Data
from torch_geometric.utils import softmax
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, lr_scheduler
from torch.nn import Sequential, Linear, ReLU, GRU, DataParallel
import wandb
import sys
os.chdir('/raid/source_tyn/3DF-GNN/code/model_clean/again/')
from model import EAFCN, EFFCN, NFTPGN, NNCFTPGN, NNCFPGN, NNCPGN
from model import GNNSCFDN, GNNDN, SCFDN, GSCFDN, NOECFP, NOMACCSDN
from loss import get_loss
from utils import regression_metrics, binary_metrics, binary_metrics_multi_target_nan
from Attention import se_block, cbam_block, eca_block, ExternalAttention, SpatialAttention

warnings.filterwarnings("ignore", category=DeprecationWarning)           
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--dataset', type=str, default='ECFP_scaffold_freesolv_3D_attr', help='breastCellLines/Bcap37_3D_attr, lipophilicity')
parser.add_argument('--num_tasks', default=1, type=int, help='sider:27,tox21:12,tocxcast:617, muv:')
parser.add_argument('--loss', default='mse', type=str, help='ce,wce,focal,bfocal...')
parser.add_argument('--gpu', default=3, type=int, help='0,1,2,3')
parser.add_argument('--mol_edge_in_dim', default=4, type=int)
parser.add_argument('--cnn_dim', default=66, type=int, help='output size of model')
parser.add_argument('--EPOCHS', default=100, type=int, help='number of epoch')
parser.add_argument('--model', default='GNNSCFDN', type=str, help='GNNSCFDN,SCFDN, GSCFDN, NOECFP, NOMACCSDN')
parser.add_argument('--fd_dim', default=64, type=int)
parser.add_argument('--out_dim', default=66, type=int)
parser.add_argument('--num_heads', default=2, type=int, help='the number of num_heads')
parser.add_argument('--weight_decay', default=1e-8, type=float, help='the nomalization parameter')
parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts', type=str, help='ReduceLROnPlateau, ExponentialLR...')
parser.add_argument('--optimizer', default='Adam', type=str, help='Adam, SGD...')
parser.add_argument('--dropout_rate', default=0.2, type=float, help='the rate of nn dropout')
parser.add_argument('--early_stop_patience', default=20, type=int, help='the rate of nn dropout')

parser.add_argument('--dim', default=16, type=int, help='the number of num_heads')
parser.add_argument('--batch', default=128, type=int, help='the nomalization parameter')
parser.add_argument('--lr', default=0.0001, type=float, help='ReduceLROnPlateau, ExponentialLR...')
parser.add_argument('--lr_reduce', default=0.1, type=float, help='Adam, SGD...')
parser.add_argument('--dp', default=0.3, type=float, help='the rate of nn dropout')
parser.add_argument('--n_iter', default=6, type=int, help='the nomalization parameter')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda')
torch.cuda.set_device(args.gpu)
config_dict = vars(args)  # vars()函数回对象object的属性和属性值的字典对象

def setup_wandb(cfg):
    config_dict = args()
    kwargs = {'name':cfg.general.name, 'project':f'{args.model}_{args.dataset}','entity':'Yanara-Tian',
              'config':config_dict, 'settings':wandb.Settings(_disable_stats=True), 'reinit':True, 
              'mode':cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def create_batch(sizes):
    batch = []
    for i, size in enumerate(sizes):
        batch.extend([i] * size)
    batch = torch.tensor(batch, dtype=torch.int64).to(device)
    return batch

def cat_multi_to_single_index(file, path):
    inters = []
    atom_matrixs = []
    fingers = []
    mol_len = []
    mol_feature = torch.FloatTensor()
    mol_edges_feature = torch.FloatTensor()
    mol_coord_feature = torch.FloatTensor()
    edge_index = torch.LongTensor()
    edge_attr = torch.FloatTensor()
    mol_nc_feature = torch.FloatTensor()
    smi_length = 0

    for fi in file:
        pt_file = os.path.join(path, fi)
        pt = torch.load(pt_file)
        if args.dataset.split('/')[0] =='breastCellLines':
            interaction = [eval(pt.interaction)]
            inters.append(torch.tensor(interaction))
        else:
            inters.append(torch.tensor(pt.interaction))
            
        atom_matrixs.append(pt.atom_matrix)
        fingers.append(pt.finger)

        mol_feature = torch.cat((mol_feature, pt.mol_feature), 0)  # 106,24
        mol_edges_feature = torch.cat((mol_edges_feature, pt.mol_edges_feature), 0) # 224, 8
        mol_coord_feature = torch.cat((mol_coord_feature, pt.mol_coord_feature), 0) # 106, 3
        mol_nc_feature = torch.cat((mol_nc_feature, pt.mol_nc_feas), 0) # 106, 3
        edge_attr = torch.cat((edge_attr, pt.edge_attr), 0)  # 100,6
        edge_index = torch.cat((edge_index, pt.edge_index+smi_length), 1) # 2, 224 
        smi_length = smi_length + pt.mol_feature.shape[0]  # 106 
        mol_len.append(pt.mol_feature.shape[0])
        
    y_true = torch.stack(inters, 0).view(1, -1).to(device)
    mol_batch = create_batch(mol_len)
    mol_feats = Data(mol_len=mol_len,
                    interaction=y_true.to(device),
                    mol_batch=mol_batch.to(device),
                    mol_feature=mol_feature.to(device),
                    mol_edges_feature=mol_edges_feature.to(device),
                    mol_coord_feature=mol_coord_feature.to(device),
                    edge_index=edge_index.to(device),
                    edge_attr=edge_attr.to(device),
                    mol_nc_feature=mol_nc_feature.to(device),
                    atom_matrixs=atom_matrixs,
                    fingers=fingers)
    
    return y_true, mol_feats

def valid_test(dataloader, path, epoch, length, dataset):
    model.eval()
    total_loss = 0
    total_y_true = torch.Tensor()
    total_y_pred = torch.Tensor()

    with torch.no_grad():
        c = 0
        for file in dataloader:
            y_true, x_feats = cat_multi_to_single_index(file, path)
            y_pred = model(x_feats).reshape(1,-1)
            c = c + args.batch
            loss = criterion(y_pred.float(), y_true.float())
            loss = loss.detach()
            total_loss += loss
            y_true = y_true.cpu()
            y_pred = y_pred.cpu()
            total_y_true = torch.cat((total_y_true, y_true), 1)
            total_y_pred = torch.cat((total_y_pred, y_pred), 1)
                        
    mean_loss = total_loss/len(dataloader)
    if args.loss == 'mse':
        results = regression_metrics(total_y_true.squeeze(), total_y_pred.squeeze())
        res = results['rmse']
    elif args.loss == 'bcel':
        total_y_true = total_y_true.reshape(-1,1)
        total_y_pred = total_y_pred.reshape(-1,1)
        results = binary_metrics(total_y_true.numpy(), total_y_pred.numpy())
        res = results['auc']

    print(f'epoch:{epoch+1}')    
    print(f'using {args.model} to valid or test {args.dataset}')
    print(f'parameters: dim:{args.dim}, n_iter:{args.n_iter}, seed:{seed}, lr:{args.lr}, \
          batch_size:{args.batch}, dropout_rate:{args.dp}, lr_reduce:{args.lr_reduce}, \
          optimizer:{args.optimizer}, loss:{args.loss}')
    print(f'{dataset}_metric:{results}', end='\n')
    print('{}: loss={:.5f}\n'.format(dataset, mean_loss))

    return mean_loss, results, res


def train_valid_test(sa_path, train_dataloader, valid_dataloader, test_dataloader, seed, lr, dp, batch):
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        optimizer = Adam(params, lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(params, lr, weight_decay=args.weight_decay)

    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduce, patience=args.lr_reduce_patience, min_lr=1e-6)
    elif args.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_loss = 10000
    best_rmse = 10
    best_auc = 0
    best_test_rmse = 100
    best_test_auc = 0
    
    with open(f"{sa_path}/{MODEL_train_NAME}.log", "a") as f:
        for epoch in range(args.EPOCHS):
            if epoch % decay_interval == 0:
                optimizer.param_groups[0]['lr'] *= args.lr_reduce

            start_t = time.time()
            model.train()
            losses = 0
            b = 0

            train_path = f'./prepro_data/{args.dataset}/train_data'
            for i, file in enumerate(train_dataloader):
                com_affinity, x_feats = cat_multi_to_single_index(file, train_path)
                output = model(x_feats).reshape(1,-1)
                loss = criterion(output.float(), com_affinity.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                loss = loss.detach()
                losses += loss.item()
                b = b + batch
                
            train_loss = losses/len(train_dataloader)
            end_t = time.time()
            spent_t = end_t - start_t
            m, s = divmod(spent_t, 60)
            h, m = divmod(m, 60)
            print(f'epoch {epoch+1} is trained by taking: {"%02d:%02d:%02d" % (h, m, s)}')
            print('total_train_loss={:.5f}\n'.format(train_loss))

            valid_path = f'./prepro_data/{args.dataset}/valid_data'
            mean_loss, results, res = valid_test(valid_dataloader, valid_path, epoch, valid_length, dataset='valid')
            save_root = f'{sa_path}/{MODEL_train_NAME}'
            os.makedirs(save_root, exist_ok=True)
            save_path = os.path.join(save_root, f'{args.model}_loss_best.pkl')
            s_path = os.path.join(save_root, f'{args.model}_rmse_best.pkl')

            test_path = f'./prepro_data/{args.dataset}/test_data'
            test_loss, test_results, test_res = valid_test(test_dataloader, test_path, epoch, test_length, dataset='test')
            ts_path = os.path.join(save_root, f'{args.model}_test_rmse_best.pkl')
            
            if mean_loss < best_loss:
                early_stop_cnt = 0
                best_loss = mean_loss
                model.cpu()
                save_dict = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'loss':best_loss}
                torch.save(save_dict, save_path) 
                model.to(device) 
            else:
                early_stop_cnt += 1
                print("early_stop_cnt", early_stop_cnt)
                
            if args.loss == 'mse':
                if res < best_rmse:
                    best_rmse = res
                    model.cpu()
                    save_dict = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'rmse':best_rmse}
                    torch.save(save_dict, s_path)
                    model.to(device)
                    best_valid_result = best_rmse
                    result = test_res

                if test_res < best_test_rmse:
                    best_test_rmse = test_res
                    model.cpu()
                    save_dict = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'test_rmse':best_rmse}
                    torch.save(save_dict, ts_path)
                    model.to(device)
                    best_test_result = best_test_rmse
                    
            elif args.loss == 'bcel':
                if res > best_auc:
                    best_auc = res
                    model.cpu()
                    save_dict = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'auc':best_auc}
                    torch.save(save_dict, s_path)
                    model.to(device)
                    best_valid_result = best_auc
                    result = test_res

                if test_res > best_test_auc:
                    best_test_auc = test_res
                    model.cpu()
                    save_dict = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'test_auc':best_auc}
                    torch.save(save_dict, ts_path)
                    model.to(device)
                    best_test_result = best_test_auc
                if 0 < args.early_stop_patience < early_stop_cnt:
                    f.write(f'Early stop hitted after the epoch {epoch} training!\n')
                    break
                
            f.write(f"{MODEL_train_NAME}_{epoch+1}_seed{seed},run_time:{h}hour{m}min{s}second,\n\
            parameters: dim:{args.dim}, n_iter:{args.n_iter}, seed:{seed}, lr:{lr}, \
                batch_size:{batch}, dropout_rate:{dp}, lr_reduce:{args.lr_reduce}\n\
            optimizer:{args.optimizer}, loss:{args.loss}, weight_decay:{args.weight_decay}, scheduler:{args.scheduler}\n\
            total_train_loss:{round(float(train_loss), 4)}\n\
            Valid:total_valid_loss:{round(float(mean_loss), 4)},metrics:{results}\n\
            Test:total_test_loss:{round(float(test_loss), 4)},metrics:{test_results}\n")
    
    return best_valid_result, result, best_test_result

def metric(value):
    value_mean = np.mean(np.array(value))
    value_std = np.std(np.array(value))
    return value_mean, value_std

log_path = f'./20240327_origin_iters/{args.model}/{args.dataset}' 
os.makedirs(log_path, exist_ok=True)
log_file_name = log_path + "/" + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
sys.stdout = Logger(log_file_name)
sys.stderr = Logger(log_file_name)
if args.dataset.split('/')[0] == 'breastCellLines':
    MODEL_train_NAME = f"{args.model}_{args.dataset.split('/')[1]}_{int(time.time())}"
else:
    MODEL_train_NAME = f"{args.model}_{int(time.time())}"

for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))

path = f'./prepro_data/{args.dataset}'
train_dataset = os.listdir(f'{path}/train_data')
valid_dataset = os.listdir(f'{path}/valid_data')
test_dataset = os.listdir(f'{path}/test_data')

train_length = len(train_dataset)
valid_length = len(valid_dataset)
test_length = len(test_dataset)

iteration = 100
decay_interval = 10
valid_rmse = []
val_test_res = []
test_rmse = []

lg_path = f'{log_path}' 
for i in range(5):
    seed = 256
    start = timeit.default_timer()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_workers = 4
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, num_workers=num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, num_workers=num_workers, shuffle=True)
    print('load data done')

    if args.model == 'GNNSCFDN':
        model = GNNSCFDN(device, args.n_iter, args.dp, in_dim=66, dim=args.dim, mol_edge_in_dim=args.mol_edge_in_dim, num_tasks=args.num_tasks)
    model.to(device)

    criterion = get_loss(f'{args.loss}')
    os.makedirs(lg_path, exist_ok=True)

    val_rmse, val_te_result, te_rmse = train_valid_test(lg_path, train_dataloader, valid_dataloader, \
        test_dataloader, seed, args.lr, args.dp, args.batch)

    end = timeit.default_timer()
    timeval = end - start
    print(f"{args.dataset}验证集在seed(256)下的最优rmse/auc值为：", val_rmse)
    print(f"{args.dataset}验证集最优时，测试集在seed(256)下的rmse/auc值为：", val_te_result)
    print(f"{args.dataset}测试集在seed(256)下的最优rmse/auc值为：", te_rmse)
    print(f"seed(256)下(每个seed100epoch)总共用时：", timeval)

    valid_rmse.append(val_rmse)
    val_test_res.append(val_te_result)
    test_rmse.append(te_rmse)

RMSE_mean_valid, RMSE_std_valid = metric(valid_rmse)
RMSE_mean_val_test, RMSE_std_val_test = metric(val_test_res)
RMSE_mean_best_test, RMSE_std_best_test = metric(test_rmse)
end = timeit.default_timer()
time = end - start

print(f"{args.dataset}验证集在固定seed下的rmse/auc值为：", valid_rmse)
print(f"{args.dataset}验证集在5个seed下的平均rmse/auc值：", RMSE_mean_valid)
print(f"{args.dataset}验证集在5个seed下的rmse/auc值的标准差：", RMSE_std_valid)

print(f"{args.dataset}验证集最优时，测试集在5个seed下的rmse/auc值为：", val_test_res)
print(f"{args.dataset}验证集最优时，测试集在5个seed下的平均rmse/auc值：", RMSE_mean_val_test)
print(f"{args.dataset}验证集最优时，测试集在5个seed下的rmse/auc值的标准差：", RMSE_std_val_test)

print(f"{args.dataset}测试集在固定seed下的最优rmse/auc值为：", test_rmse)
print(f"{args.dataset}测试集在5个seed下的最优rmse/auc的平均值：", RMSE_mean_best_test)
print(f"{args.dataset}测试集在5个seed下的最优rmse/auc值的标准差：", RMSE_std_best_test)
print(f"5个seed下(每个seed50epoch)总共用时：", time)
