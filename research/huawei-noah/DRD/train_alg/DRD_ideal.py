from select import EPOLL_CLOEXEC
# import torch
import pandas as pd
import numpy as np
# from torch.nn import BCELoss,BCEWithLogitsLoss, MSELoss
# from torch.utils.data import DataLoader, Dataset
# from torch.optim import Adam,Adadelta,RMSprop,SGD
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import mindspore
from mindspore import nn
from mindspore.common.initializer import Normal,XavierNormal, Initializer,initializer
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import transforms
import sys
import time
import itertools
sys.path.append("..")
from models.base_learn_alg import Learning_Alg
from utils.arg_parser import config_param_parser
from utils.data_wrapper import Wrap_Dataset_point1
# from utils.get_sparse import get_sparse_feature
from utils.early_stop import EarlyStopping
from utils.trans_format import format_trans
from models.evaluate import Test_Evaluator
from models.loss_func import DRD_bias_BCELoss, DRD_rel_BCELoss
from models.toy_model import rank_model, bias_model, rank_model_linear

class DRD_ideal(Learning_Alg):
    def __init__(self, args):
        super(DRD_ideal, self).__init__(args)
        
    def train(self):
        mindspore.set_context(device_target="CPU")
        t0 = time.time()
        self.test_tool, self.eval_tool, self.in_size = self._load_test_and_get_tool()
        t1 = time.time()
        input_train_loader = self._load_and_wrap_train()
        t2 = time.time()
        print('load test&vali time:', t1-t0)
        print('load train time:', t2-t1)
        self.model, self.optim, self.scheduler, self.early_stopping, self.p_model, self.p_optim = self._init_train_env()
        self._train_iteration(input_train_loader)
        super(DRD_ideal, self)._test_and_save()


    def _load_test_and_get_tool(self):
        test_dat = pd.read_json(self.fin + 'json_file/Test.json')
        test_dat = format_trans(test_dat,mode='test')

        eval_dat = pd.read_json(self.fin + 'json_file/Vali.json')
        eval_dat = format_trans(eval_dat,mode='test')

        test_tool = Test_Evaluator(test_dat, 
                                   self.eval_positions, 
                                   use_cuda=self.use_cuda)
        # eval_tool = Test_Evaluator(eval_dat, 
        #                            self.eval_positions, 
        #                            use_cuda=self.use_cuda)
        eval_tool = Test_Evaluator(test_dat, 
                                   self.eval_positions, 
                                   use_cuda=self.use_cuda)
        in_size = len(test_dat['feature'][0])

        return test_tool, eval_tool, in_size

    
    # def _train_one_batch_rel(self, batch, epoch):
    #     self.model.train()
    #     self.optim.zero_grad()
    #     BCELossfunc = DRD_rel_BCELoss(self.alpha, 0)
    #     output_score =  self.model(batch[0]).view(batch[0].size(0))
    #     trust_posi_p = batch[2]
    #     trust_nega_p = batch[3]
    #     target = batch[1]
    #     train_loss = BCELossfunc(output_score, trust_posi_p, trust_nega_p, target)
    #     train_loss.backward()
    #     self.optim.step()

    #     return train_loss
    
    def _train_one_batch_rel(self, feature, target, p_posi, p_nega):
        # Define forward function
        def forward_fn(feature, target, p_posi, p_nega):
            BCELossfunc = DRD_rel_BCELoss(self.alpha, 0)
            # logits = self.model(feature).view(feature.size)
            logits = self.model(feature).view(-1)
            loss = BCELossfunc(logits, p_posi, p_nega, target)
            return loss, logits

        # Get gradient function
        grad_fn = mindspore.value_and_grad(forward_fn, None, self.optim.parameters, has_aux=True)

        # one-step training
        (train_loss, _), grads = grad_fn(feature, target, p_posi, p_nega)
        self.optim(grads)

        return train_loss


    def _train_iteration(self, input_train_loader):
        # strat training
        dur = []

        for epoch in range(self.epoch):
            if epoch >= 0:
                t0 = time.time()
            
            loss_log = []

            self.model.set_train()
            self.p_model.set_train()

            # for _id, batch in enumerate(input_train_loader):
            for batch, (feature, target, p_posi, p_nega) in enumerate(input_train_loader.create_tuple_iterator()):
                # feature = mindspore.Tensor(feature, dtype=mindspore.float32)
                # target = mindspore.Tensor(target, dtype=mindspore.float32)
                # p_posi = mindspore.Tensor(p_posi, dtype=mindspore.float32)
                # p_nega = mindspore.Tensor(p_nega, dtype=mindspore.float32)
                train_loss = self._train_one_batch_rel(feature, target, p_posi, p_nega)
                loss_log.append(train_loss.asnumpy())

            if self.eval_tag:
                test_result = self.eval_tool.evaluate(self.model)
                
                ndcg_tst = test_result['NDCG'][self.topK]
                mrr_tst = test_result['MRR'][self.topK]
                precision_tst = test_result['Precision'][self.topK]
                map_tst = test_result['MAP'][self.topK]
                ndcg_full_tst = test_result['NDCG_full'][self.topK]

                self.early_stopping(ndcg_tst*(-1), self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break 
                
                if epoch >= 0:
                    dur.append(time.time() - t0)
                
                print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Vali_NDCG@10 {:.4f} | "
                        "Vali_MRR@10 {:.4f}| Vali_Precision@10 {:.4f} | Vali_MAP@10 {:.4f} | Vali_NDCG_full {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),ndcg_tst,
                                                        mrr_tst, precision_tst, map_tst, ndcg_full_tst))
            else:
                # torch.save(self.model.state_dict(), '{}_checkpoint.pt'.format(self.fout))
                mindspore.save_checkpoint(self.model, '{}_checkpoint.ckpt'.format(self.fout))

                if epoch >= 0:
                    dur.append(time.time() - t0)
                
                print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} |".format(epoch, np.mean(dur), np.mean(loss_log)))                           


    def _init_train_env(self):
        model = rank_model(self.in_size, self.hidden_size, self.drop_out)
        p_model = bias_model(self.topK, self.hidden_size, self.drop_out)
        
        for name, param in model.parameters_and_names():
            if 'weight' in name:
                param.set_data(initializer(XavierNormal(), param.shape, param.dtype))
            # if 'bias' in name:
            #     param.set_data(initializer(XavierNormal(), param.shape, param.dtype))

        for name, param in p_model.parameters_and_names():
            if 'weight' in name:
                param.set_data(initializer(XavierNormal(), param.shape, param.dtype))
            # if 'bias' in name:
            #     param.set_data(initializer(XavierNormal(), param.shape, param.dtype))
        

        # if self.use_cuda:
        #     #model = nn.DataParallel(model)
        #     model = model.cuda()
        #     p_model = p_model.cuda()
        
        if self.optimizer == 'adam':
            optim = nn.Adam(model.trainable_params(), learning_rate=self.lr, weight_decay=self.weight_decay)
            p_optim = nn.Adam(p_model.trainable_params(), learning_rate=self.lr, weight_decay=self.weight_decay)
            # optim = Adam(itertools.chain(model.parameters(),p_model.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optim = nn.SGD(model.trainable_params(), learning_rate=self.lr, weight_decay=self.weight_decay)
            p_optim = nn.SGD(p_model.trainable_params(), learning_rate=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer== 'adadelta':
            optim = nn.Adadelta(model.trainable_params(), learning_rate=self.lr, weight_decay=self.weight_decay)
            p_optim = nn.Adadelta(p_model.trainable_params(), learning_rate=self.lr, weight_decay=self.weight_decay)

        # scheduler = ReduceLROnPlateau(optim, 
        #                               patience=10, 
        #                               mode=self.schedul_mode,
        #                               threshold=1e-6,
        #                               verbose=True)
        scheduler = 0
        early_stopping = EarlyStopping(self.fout, self.patience, verbose=True) 

        return model, optim, scheduler, early_stopping, p_model, p_optim



    def _load_and_wrap_train(self):
        if self.pairwise:
            train_log_fe = pd.read_json(self.fin + 'click_log/Train_log_trans.json')

            if self.topK > 0:
                train_log_fe['isSelect'] = train_log_fe['rankPosition'].apply(lambda x: 1 if x<self.topK else 0)

            train_log_fe = train_log_fe[train_log_fe['isSelect']==1] # For Reweight method, we just use selected data!!!


            eyetracking_examProb = [0.68, 0.61, 0.48, 0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06, 0.00]
            eyetracking_examProb = [x**self.eta for x in eyetracking_examProb]
            train_log_fe['eps_posi'] = train_log_fe['rankPosition'].apply(lambda x: (98-x)/100)
            train_log_fe['eps_nega'] = train_log_fe['rankPosition'].apply(lambda x: self.init_prob/(x+1))
            train_log_fe['theta'] = train_log_fe['rankPosition'].apply(lambda x: eyetracking_examProb[x if x < self.topK else -1])
            train_log_fe['p_posi'] = train_log_fe.apply(lambda row: row['eps_posi']*row['theta'],axis=1)
            train_log_fe['p_nega'] = train_log_fe.apply(lambda row: row['eps_nega']*row['theta'],axis=1)
            

            input_train = Wrap_Dataset_point1(train_log_fe['p_posi'].to_list(),
                                            train_log_fe['p_nega'].to_list(),
                                            train_log_fe['Click'].to_list(),
                                            train_log_fe['feature'].to_list())
            # input_train_loader = DataLoader(input_train, 
            #                                 batch_size=self.batch_size, 
            #                                 shuffle=True)
            input_train_dat = GeneratorDataset(source=input_train, 
                                               column_names=['feature','target','p_posi','p_nega'])
            # column_types=[mindspore.float32,mindspore.float32,mindspore.float32,mindspore.float32]

            # val_transform = transforms.TypeCast(mindspore.float32)
            # input_train_dat = input_train_dat.map(val_transform, 'feature')
            # input_train_dat = input_train_dat.map(val_transform, 'target')
            # input_train_dat = input_train_dat.map(val_transform, 'p_posi')
            # input_train_dat = input_train_dat.map(val_transform, 'p_nega')
            input_train_loader = input_train_dat.batch(batch_size=self.batch_size)
            input_train_loader.shuffle(64)
        return input_train_loader


    def _get_onehotvec(self, idx, vec_len):
        zero_pad = [0]*vec_len
        zero_pad[idx] = 1
        return zero_pad

if __name__=="__main__":
    print('Start ...')
    parser = config_param_parser()
    args = parser.parse_args()
    learner = DRD_ideal(args)
    learner.train()