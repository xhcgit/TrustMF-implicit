import torch as t
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, userNum, itemNum, hide_dim):
        super(Model, self).__init__()
        self.B = nn.Embedding(userNum, hide_dim)
        self.V = nn.Embedding(itemNum, hide_dim)
        self.W = nn.Embedding(userNum, hide_dim)
        nn.init.xavier_normal_(self.B.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.W.weight)
        self.hide_dim = hide_dim
    
    def test(self, uid, iid):
        b = self.B(uid)
        v = self.V(iid)
        pred_r = t.sum(b*v, dim=1)
        return pred_r

        
    def forward(self, trainMat, trustMat, uid):
        # uid_t = t.from_numpy(uid).long().cuda()
        user_idx = uid[trainMat[uid].tocoo().row]
        item_idx = trainMat[uid].tocoo().col

        user_idx2 = uid[trustMat[uid].tocoo().row]
        trust_idx2 = trustMat[uid].tocoo().col

        u_set = t.from_numpy(np.unique(user_idx)).long().cuda()
        i_set = t.from_numpy(np.unique(item_idx)).long().cuda()
        t_set = t.from_numpy(np.unique(trust_idx2)).long().cuda()

        b = self.B(t.from_numpy(user_idx).long().cuda())
        v = self.V(t.from_numpy(item_idx).long().cuda())

        b2 = self.B(t.from_numpy(user_idx2).long().cuda())
        w = self.W(t.from_numpy(trust_idx2).long().cuda())

        pred_r = t.sigmoid(t.sum(b*v, dim=1))
        pred_t = t.sigmoid(t.sum(b2*w, dim=1))

        # regLoss1 = t.sum(
        #     t.sum(self.B(u_set)*self.B(u_set), dim=1, keepdim=True) * \
        #     ( t.from_numpy(np.sum(trainMat[u_set.cpu().numpy()] != 0, axis=1)).float().cuda() + \
        #         t.from_numpy(np.sum(trustMat[u_set.cpu().numpy()], axis=1)).float().cuda()))
        # regLoss2 = t.sum(
        #     t.sum(self.V(i_set)*self.V(i_set), dim=1, keepdim=True) *\
        #     t.from_numpy(np.sum(trainMat[:, i_set.cpu().numpy()]!=0, axis=0).reshape(-1, 1)).float().cuda())
        # regLoss3 = t.sum(
        #     t.sum(self.W(t_set)*self.W(t_set), axis=1, keepdim=True) *\
        #     t.from_numpy(np.sum(trustMat[:, t_set], axis=0).reshape(-1, 1).astype(np.float64)).float().cuda())

        # regLoss = regLoss1 + regLoss2 + regLoss3
        
        return pred_r, pred_t#, regLoss




        

