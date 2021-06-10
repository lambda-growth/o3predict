import torch
from torch import nn
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import argparse
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# pd.set_option('display.max_columns', None)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def model_parameters_init(model):
    '''
    kaiming init
    '''
    for p in model.parameters():
        if len(p.shape) >= 2:
            nn.init.kaiming_normal_(p)
    return model


class NetWork(nn.Module):
    '''
    只输入一个aqi，其他置0
    '''

    def __init__(self):
        super(NetWork, self).__init__()
        self.lstm = nn.LSTM(17, 128, 1, batch_first=True)
        self.linear_1 = nn.Linear(128, 10)
        self.linear_2 = nn.Linear(10, 1)
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(13)

    def forward(self, x):
        x = self.bn(x)
        x, (h_1, c_1) = self.lstm(x)
        x = x[:, 10:, :]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = x.squeeze(2)
        return x



def train(args, train_loader, valid_loader, model, criterion, optimizer, scheduler, device):
    # save model or not
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epochs = args.epochs
    train_losses = []
    valid_losses = []
    for epoch_id in range(epochs):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        ######################
        # training the model#
        ######################
        train_batch_cnt = 0
        for batch_idx, batch in enumerate(train_loader):
            train_batch_cnt += 1
            x = batch['x']
            y = batch['y']

            # groundtruth
            x = x.to(device)
            y = y.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get out_puts
            pred_y = model(x)

            # get loss
            loss = criterion(y, pred_y)
            train_loss += loss.item()

            # do bp
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]  MSELoss: {:.6f}'.format(
                    epoch_id,
                    batch_idx * len(x),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()
                )
                )
        # 记录train_loss
        train_loss /= train_batch_cnt
        train_losses.append(train_loss)

        ######################
        # validate the model #
        ######################
        valid_loss = 0.0
        # change model mode to eval ,not use BN/Dropout
        model.eval()
        with torch.no_grad():
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                x = batch['x']
                y = batch['y']

                x = x.to(device)
                y = y.to(device)

                pred_y = model(x)

                valid_loss_batch = criterion(y, pred_y)
                valid_loss += valid_loss_batch.item()

            valid_loss /= valid_batch_cnt * 1.0
            # 记录valid_loss
            valid_losses.append(valid_loss)
            print('Valid: MSELoss: {:.6f}'.format(valid_loss))
        # 学习率衰减
        scheduler.step()
        print('===========================================================')
        # save model
        if args.save_model and epoch_id % 10 == 0:
            saved_model_name = os.path.join(args.save_directory, 'epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    return train_losses, valid_losses


def test(args, valid_loader, model, criterion, device):
    path_model = os.path.join(args.save_directory, 'epoch' + '_' + str(args.number) + '.pt')
    model.load_state_dict(torch.load(path_model))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        valid_batch_cnt = 0
        valid_loss = 0
        mb_y = 0
        mb_pred_y = 0
        for valid_batch_idx, batch in enumerate(valid_loader):
            valid_batch_cnt += 1
            x = batch['x']
            y = batch['y']

            x = x.to(device)
            y = y.to(device)
            pred_y = model(x)

            mb_y += y.mean()
            mb_pred_y += pred_y.mean()

            valid_loss_batch = criterion(y, pred_y)
            valid_loss += valid_loss_batch.item()

        valid_loss /= valid_batch_cnt * 1.0
        print('Valid: MSELoss: {:.6f}'.format(valid_loss))
        print('MB_Y: {:.2f}, MB_PredY: {:.2f}'.format(mb_y.cpu().numpy(), mb_pred_y.cpu().numpy()))

def create_dataset(data,n_predictions,n_next):
    '''
    n_predictions:前面步数
    n_next: 预测后面步数
    对数据进行处理
    '''
    train_X, train_Y = [], []
    for i in range(data.shape[0]-n_predictions-n_next-1):
        a = data[i:(i+n_predictions),:]
        train_X.append(a)
        tempb = data[(i+n_predictions):(i+n_predictions+n_next),0]
        b = []
        for j in range(len(tempb)):
            b.append(tempb[j])
        train_Y.append(b)
    train_X = np.array(train_X,dtype='float64')
    train_Y = np.array(train_Y,dtype='float64')

    return train_X, train_Y


def main(args):
    #设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #设置CPU/GPU
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    ###############################################################################################################
    print('===> Loading Datasets')
    #读数据
    data = pd.read_csv('./data/2020汕头日.csv')
    print(data.columns)
    data.drop_duplicates(subset=['time'], inplace=True)
    data.dropna(axis=0, subset=['aqi', 'cityname', 'co', 'no2', 'so2', 'o3_8h',
                                'pm2_5', 'pm10', 'wda_max', 'wda_mean', 'wda_min', 'temp_max',
                                'temp_mean', 'temp_min', 'dswrf_max', 'dswrf_mean', 'dswrf_min',
                                'humi_max', 'humi_mean', 'humi_min', 'apcp_max', 'apcp_mean',
                                'apcp_min'], inplace=True)
    for idx, city in enumerate(data['cityname'].unique()):
        data1 = data.drop(data[data.cityname != city].index)
        print(len(data1))
        data1 = data1.loc[:, ['aqi', 'co', 'no2', 'so2', 'o3_8h',
                              'pm2_5', 'pm10', 'wda_max', 'wda_mean', 'wda_min', 'temp_max',
                              'temp_mean', 'temp_min', 'dswrf_max', 'dswrf_mean', 'dswrf_min',
                              'humi_max', 'humi_mean', 'humi_min', 'apcp_max', 'apcp_mean',
                              'apcp_min']]
        if idx == 0:
            training_set = data1.iloc[1:-300, :].values
            test_set = data1.iloc[-300:, :].values
        else:
            training_set2 = data1.iloc[1:, :].values
            training_set = np.append(training_set, training_set2, axis=0)

    # 归一化

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range, np.max(data), np.min(data)

    def transform(data):
        return (data - data_min) / (data_max - data_min)

    def inverse_transform(data):
        return data * (data_max - data_min) + data_min

    sc = StandardScaler()  # 定义归一化：归一化到(0，1)之间
    training_set_scaled, data_max, data_min = normalization(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
    test_set_scaled = transform(test_set)

    train_X, train_Y = create_dataset(training_set_scaled, 10, 1)

    # 将训练集由list格式变为array格式
    train_X, train_Y = np.array(train_X), np.array(train_Y)
    test_X, test_Y = create_dataset(test_set_scaled, 10, 1)
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
    test_X, test_Y = torch.from_numpy(test_X).float(), torch.from_numpy(test_Y).float()
    train_dataset = TensorDataset(train_X, train_Y)
    test_dataset = TensorDataset(test_X, test_Y)
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size=32,
                              shuffle=True)
    test_loader = DataLoader(dataset = test_dataset,
                              batch_size=32,
                              shuffle=True)

    # predict_loader = torch.utils.data.DataLoader(val_set, batch_size=args.predict_batch_size, num_workers= 0, pin_memory= False)
    ###############################################################################################################
    print('===> Building Model')
    print('===> runing on {}'.format(device))
    ###############################################################################################################
    print('===> init model')
    model = NetWork()
    ###############################################################################################################
    model.to(device)
    criterion = nn.MSELoss()
#     criterion = nn.SmoothL1Loss()
#     optimizer = optim.Adam(model.parameters(), lr= args.lr)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    #学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1 , 0.85)
    ###############################################################################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses = train(args, train_loader, test_loader, model, criterion, optimizer, scheduler, device)
        print('===> Done!')
        return train_losses, valid_losses

    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        test(args, test_loader, model, criterion, device)
        print('===> Done!')
        return None, None
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        path_model = os.path.join(args.save_directory, 'epoch' + '_' + str(args.number) + '.pt')
        model.load_state_dict(torch.load(path_model))
        model = model.to(device)
        train_losses, valid_losses = train(args, train_loader, test_loader, model, criterion, optimizer, scheduler, device)
        print('===> Done!')
        return train_losses, valid_losses

    # elif args.phase == 'Predict' or args.phase == 'predict':
    #     print('===> Predict')
    #     predict(args, val_set, model, device)
    #     print('===> Done!')
    #     return None, None


def predict(args, val_set, model, device):
    path_model = os.path.join(args.save_directory, 'epoch' + '_' + str(args.number) + '.pt')
    model.load_state_dict(torch.load(path_model))
    model = model.to(device)
    model.eval()
    data = val_set[args.idx]
    with torch.no_grad():
        x = data['x'].to(device).unsqueeze(0)
        y = data['y'].numpy()
        pred_y = model(x)
        pred_y = pred_y.cpu().numpy()
        plt.figure(0,(8,6))
        plt.plot(range(len(y)),y,'b-o')
        plt.plot(range(len(y)),pred_y[0],'r-o')
        plt.xlabel('Following 72 Hours')
        plt.ylabel('AQI')
        plt.legend(['obs','predict'])
        plt.show()
        plt.savefig('pic.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--predict_batch_size', type=int, default=1, metavar='N',
                        help='input batch size for predict (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='save the current Model')
    parser.add_argument('--save_directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    parser.add_argument('--number', type=int, default=0,
                        help='which model to use')
    parser.add_argument('--idx', type=int, default=0,
                        help='which sample to predict')
    args = parser.parse_args(['--batch_size=8',
                              '--test_batch_size=100',
                              '--predict_batch_size=1',
                              '--epochs=100',
                              '--lr=0.00001',
                              '--momentum=0.5',
                              '--cuda',
                              '--seed=1',
                              '--log_interval=50',
                              '--save_model',
                              '--save_directory=trained_models',
                              '--number=80',
                              '--idx=10',
                              '--phase=Train'])
    ##############################################################################################################
    start = time.time()
    train_losses, valid_losses = main(args)
    end = time.time()
    print('耗时：{}s'.format(end - start))
    torch.cuda.empty_cache()
