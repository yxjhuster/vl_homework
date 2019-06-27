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

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=1, writer='../tensorboardx'):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 150  # Steps

        self._train_dataset_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=self._custom_collate)

        self._val_dataset_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=self._custom_collate)
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        print('create events!')
        logdir = os.path.join(writer,
                              datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print(logdir)
        self.writer = SummaryWriter(logdir)

        print(self._model)

    def _custom_collate(self, batch):
        new_batch = {}
        new_batch.update({'feature': torch.stack(
            [small_batch['feature'] for small_batch in batch])})
        new_batch.update(
            {'question': [small_batch['question'] for small_batch in batch]})
        new_batch.update({'gt': torch.stack(
            [small_batch['gt'] for small_batch in batch])})
        new_batch.update({'mask': torch.stack(
            [small_batch['mask'] for small_batch in batch])})
        return new_batch

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
                question_indx = batch_data['question']
                question = [self.create_one_hot(idx) for idx in question_indx]
                question = [ques.cuda() for ques in question]
                mask = batch_data['mask'].cuda()
                predicted_answer = self._model(feature, question, mask)
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
    #         output = nn.Softmax()(predicted_answer)
    #         output = output.data.cpu().numpy()
    #         # pdb.set_trace()
    #         gt = batch_data['gt'].numpy()
    #         batch_size = len(gt)
    #         corr = 0
    #         for i in range(batch_size):
    #             max_idx = np.argmax(output[i])
    #             if(max_idx == gt[i]):
    #                 corr = corr + 1
    #         # pdb.set_trace()
    #         acc += corr / 100.
    #         if batch_id > 50:
    #             break

    #     return acc / 50.

    def create_one_hot(self, index_list):
        bag = []
        for idx in index_list:
            one_hot = np.zeros(2000)
            one_hot[idx] = 1
            bag.append(one_hot)
        bag = np.array(bag)
        return torch.tensor(bag, dtype=torch.float32)

    def train(self):
        self._model.train()

        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                feature = batch_data['feature'].cuda()
                question_indx = batch_data['question']
                question = [self.create_one_hot(idx) for idx in question_indx]
                question = [ques.cuda() for ques in question]
                mask = batch_data['mask'].cuda()
                # pdb.set_trace()
                predicted_answer = self._model(feature, question, mask)
                # pdb.set_trace()
                ground_truth_answer = batch_data['gt'].cuda()
                # pdb.set_trace()
                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)
                # print(loss)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch,
                                                                      batch_id, num_batches, loss))
                    # pdb.set_trace()
                    # TODO: you probably want to plot something here
                    self.writer.add_scalar(
                        'training/loss', loss.item(), current_step)

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate()
                    self._model.train()
                    print("current_step: {} has val accuracy {}".format(
                        current_step, val_accuracy))
                    # TODO: you probably want to plot something here
                    self.writer.add_scalar(
                        'validating/mAP', val_accuracy, current_step)
