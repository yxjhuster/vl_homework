from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime
import sklearn.metrics
import math
import os
import pdb


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, writer='../tensorboardx'):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 150  # Steps

        self._train_dataset_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)
        self._val_dataset_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        # print(self._model)

        print('create events!')
        logdir = os.path.join(writer,
                              datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print(logdir)
        self.writer = SummaryWriter(logdir)

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        # TODO. Should return your validation accuracy
        ap_list = []
        with torch.no_grad():
            for batch_id, batch_data in enumerate(self._val_dataset_loader):
                feature = batch_data['feature'].cuda()
                question = batch_data['question'].cuda()
                predicted_answer = self._model(feature, question)
                output = nn.Softmax(dim=1)(predicted_answer)
                output = output.data.cpu().numpy()
                # pdb.set_trace()
                gt = batch_data['gt']
                batch_size = len(gt)
                target = []
                for i in range(batch_size):
                    gt_cls = np.zeros(2001)
                    gt_cls[gt[i]] = 1
                    target.append(gt_cls)
                target = np.array(target)
                mAP = self.metric1(output, target)
                ap_list.append(mAP)
                if batch_id > 50:
                    break

        return np.sum(ap_list) / len(ap_list)

    def metric1(self, output, target):
        batch_size = output.shape[0]
        output_val = output
        target_val = target
        AP = []
        for i in range(batch_size):
            pred_cls = output_val[i]
            gt_cls = target_val[i]

            ap = sklearn.metrics.average_precision_score(
                gt_cls, pred_cls, average=None)
            if math.isnan(ap):
                ap = 0
            AP.append(ap)
        mAP = np.nanmean(AP)
        return mAP

    # def validate(self):
    #     # TODO. Should return your validation accuracy
    #     acc = 0.
    #     for batch_id, batch_data in enumerate(self._val_dataset_loader):
    #         feature = batch_data['feature'].cuda()
    #         question = batch_data['question'].cuda()
    #         predicted_answer = self._model(feature, question)
    #         output = nn.Softmax(dim=1)(predicted_answer)
    #         output = output.data.cpu().numpy()
    #         # pdb.set_trace()
    #         gt = batch_data['gt'].numpy()
    #         batch_size = len(gt)
    #         corr = 0
    #         for i in range(batch_size):
    #             max_idx = np.argmax(output[i])
    #             # print('{},{}'.format(max_idx,gt[i]))
    #             if(max_idx == gt[i]): 
    #                 corr = corr + 1
    #         # pdb.set_trace()
    #         acc += corr/100.
    #         if batch_id > 20:
    #             break

    #     return acc / 20.

    # def validate(self):
    # # TODO. Should return your validation accuracy
    #     ap_list = []
    #     for batch_id, batch_data in enumerate(self._val_dataset_loader):
    #     # for batch_id, batch_data in enumerate(self._train_dataset_loader):
    #         feature = batch_data['feature'].cuda()
    #         question = batch_data['question'].cuda()
    #         with torch.no_grad():
    #             predicted_answer = self._model(feature, question)
    #         output = nn.Softmax(dim=0)(predicted_answer)
    #         output = output.data.cpu().numpy()
    #         gt = batch_data['gt']
    #         batch_size = len(gt)
    #         n_class = output.shape[1]
    #         # use top 5
    #         AP = 0.0
    #         for i in range(batch_size):
    #             pred_cls = output[i]
    #             gt_cls = np.zeros(2001)
    #             gt_cls[gt[i]] = 1

    #             idx = sorted(range(n_class),
    #                          key=lambda j: pred_cls[j], reverse=True)

    #             tmp = 0.0
    #             for k in range(5):
    #                 tmp += pred_cls[idx[k]] * gt_cls[idx[k]]
    #             if tmp != 0:
    #                 AP += 1.0
    #         mAP = AP / batch_size
    #         ap_list.append(mAP)
    #         if batch_id > 20: break
            
    #     return np.sum(ap_list) / len(ap_list)

    def train(self):
        self._model.train()

        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                feature = batch_data['feature'].cuda()
                question = batch_data['question'].cuda()
                predicted_answer = self._model(feature, question)
                ground_truth_answer = batch_data['gt'].cuda()
                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch,
                                                                      batch_id, num_batches, loss))
                    # TODO: you probably want to plot something here
                    self.writer.add_scalar(
                        'training/loss', loss.item(), current_step)
                    # output = nn.Softmax(dim=0)(predicted_answer)
                    # output = output.data.cpu().numpy()
                    # gt = ground_truth_answer.cpu().numpy()
                    # target = []
                    # for i in range(100):
                    #     gt_cls = np.zeros(2001)
                    #     gt_cls[gt[i]] = 1
                    #     target.append(gt_cls)
                    # mAP = self.metric1(output, target)
                    # print("Epoch: {}, Batch {}/{} has mAP {}".format(epoch,
                                                                     # batch_id, num_batches, mAP))

                if current_step % self._test_freq == 0 and current_step != 0:
                    self._model.eval()
                    val_accuracy=self.validate()
                    self._model.train()
                    print("current_step: {} has val accuracy {}".format(
                        current_step, val_accuracy))
                    # TODO: you probably want to plot something here
                    self.writer.add_scalar(
                        'validating/mAP', val_accuracy, current_step)
