# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ToolScripts.TimeLogger import log
import pickle
import os
import sys
import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import time
from model import Model
import argparse
import time
from process import loadData
from BPRData import BPRData
import evaluate
import torch.utils.data as dataloader
modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")


class TrustMF():

    def getData(self, args):
        data = loadData(args.dataset, args.cv)
        trainMat, trustMat, testData = data
        return trainMat, trustMat, testData

    def __init__(self, args):
        self.args = args
        self.datasetDir = os.path.join(os.path.dirname(os.getcwd()), "dataset", args.dataset, 'implicit', "cv{0}".format(args.cv))

        trainMat, trustMat, testData = self.getData(args)
        trainMat = (trainMat !=0)*1

        self.trustMat = trustMat
        self.trainMat = trainMat
        self.trainMask  = (self.trainMat != 0)
        self.userNum, self.itemNum = self.trainMat.shape
        self.hide_dim = self.args.hide_dim
        self.loss_rmse = nn.MSELoss(reduction='sum')#不求平均
        self.curEpoch = 0
        test_dataset = BPRData(testData, self.itemNum, self.trainMat, 0, False)
        self.test_loader  = dataloader.DataLoader(test_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)

    #初始化参数
    def prepareModel(self):
        np.random.seed(self.args.seed)
        t.manual_seed(self.args.seed)
        t.cuda.manual_seed(self.args.seed)

        self.model = Model(self.userNum, self.itemNum, self.hide_dim).cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.reg)

    def run(self):
        #判断是导入模型还是重新训练模型
        self.prepareModel()
        cvWait = 0
        best_HR = 0.1
        for e in range(self.curEpoch, self.args.epochs+1):
            self.curEpoch = e
            log("**************************************************************")
            epoch_loss = self.trainModel(self.trainMat, self.trustMat)


            #验证
            if e>12:
                HR, NDCG = self.testModel()
                log("epoch %d/%d, test hr=%.4f, test ndcg=%.4f"%(e, self.args.epochs, HR, NDCG))
            else:
                HR, NDCG = 0, 0
                cvWait = 0

            if HR > best_HR:
                best_HR = HR
                cvWait = 0
            else:
                cvWait += 1
                log("cvWait = %d"%(cvWait))

            if cvWait == 5:
                HR, NDCG = self.testModel(save=True)
                with open(self.datasetDir + "/test_data.csv".format(self.args.cv), 'rb') as fs:
                    test_data = pickle.load(fs)
                uids = np.array(test_data[::101])[:,0]
                data = {}
                assert len(uids) == len(HR)
                assert len(uids) == len(np.unique(uids))
                for i in range(len(uids)):
                    uid = uids[i]
                    data[uid] = [HR[i], NDCG[i]]

                with open("TrustMF-{0}-cv{1}-test.pkl".format(self.args.dataset, self.args.cv), 'wb') as fs:
                    pickle.dump(data, fs)
                break

    

    def trainModel(self, trainMat, trustMat):
        batch = self.args.batch
        num = trainMat.shape[0]
        shuffledIds = np.random.permutation(num)
        steps = int(np.ceil(num / batch))
        epoch_loss = 0
        for i in range(steps):
            ed = min((i+1) * batch, num)
            batch_ids = shuffledIds[i * batch: ed]
            user_idx = batch_ids
            label_r = t.from_numpy(trainMat[user_idx].data).float().to(device_gpu)
            label_t = t.from_numpy(trustMat[user_idx].data).float().to(device_gpu)

            # pred_r, pred_t, regLoss = self.model(trainMat, trustMat, user_idx)
            pred_r, pred_t = self.model(trainMat, trustMat, user_idx)
            
            loss_r = self.loss_rmse(pred_r.view(-1), label_r.view(-1))
            loss_t = self.loss_rmse(pred_t.view(-1), label_t.view(-1))
            
            loss = loss_r/trainMat[user_idx].nnz + loss_t/trustMat[user_idx].nnz
            epoch_loss += loss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('setp %d/%d, step_loss = %.4f'%(i,steps, loss_r.item()), save=False, oneline=True)
        return epoch_loss


    def testModel(self, save=False):
        HR, NDCG = [], []
        for user, item_i in self.test_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()

            pred_i = self.model.test(user, item_i)
            batch = int(user.cpu().numpy().size/101)

            assert user.cpu().numpy().size % 101 ==0
            for i in range(batch):
                batch_scores = pred_i[i*101: (i+1)*101].view(-1)
                _, indices = t.topk(batch_scores, self.args.top_k)
                tmp_item_i = item_i[i*101: (i+1)*101]
                recommends = t.take(tmp_item_i, indices).cpu().numpy().tolist()
                gt_item = tmp_item_i[0].item()
                HR.append(evaluate.hit(gt_item, recommends))
                NDCG.append(evaluate.ndcg(gt_item, recommends))
        if save:
            return HR, NDCG
        else:
            return np.mean(HR), np.mean(NDCG)
            

    def getModelName(self):
        title = "TrustMF_"
        ModelName = title + dataset + "_" + modelUTCStr + \
        "_CV" + str(self.args.cv) + \
        "_reg_" + str(self.args.reg)+ \
        "_hide_" + str(self.hide_dim) + \
        "_batch_" + str(self.args.batch) + \
        "_lr_" + str(self.args.lr)
        return ModelName



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR-GMI main.py')
    parser.add_argument('--reg', type=float, default=0, metavar='N', help='reg weight')
    # parser.add_argument('--reg_t', type=float, default=0.5, metavar='N', help='reg weight')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--batch', type=int, default=64, metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch', type=int, default=1024, metavar='N', help='input batch size for training')
    parser.add_argument('--hide_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')

    parser.add_argument('--dataset', type=str, default="Epinions_time")
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--seed', type=int, default=29, metavar='int', help='random seed')
    parser.add_argument('--top_k', type=int, default=10)

    args = parser.parse_args()
    dataset = args.dataset


    hope = TrustMF(args)

    modelName = hope.getModelName()
    
    print('ModelNmae = ' + modelName)

    hope.run()

