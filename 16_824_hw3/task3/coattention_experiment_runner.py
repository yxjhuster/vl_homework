import sys
sys.path.append('../')
from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
from external.vqa.vqa import VQA

import torch
import torch.nn as nn

import pdb


class CoattentionNetExperimentRunner(ExperimentRunnerBase):
  """
  Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
  """

  def __init__(self, train_image_dir, train_question_path, train_annotation_path,
               test_image_dir, test_question_path, test_annotation_path, batch_size, num_epochs,
               num_data_loader_workers):

    self.vqa_loader = VQA(annotation_file=train_annotation_path,
                          question_file=train_question_path)

    self.entries = self.vqa_loader.qqa
    self.qa = self.vqa_loader.qa

    bag_word_question = self.get_bag_of_word_question()
    bag_word_answer = self.get_bag_of_word_answer()

    train_dataset = VqaDataset(image_dir=train_image_dir,
                               question_json_file_path=train_question_path,
                               annotation_json_file_path=train_annotation_path,
                               image_filename_pattern="COCO_train2014_{}.jpg",
                               bag_word_question=bag_word_question,
                               bag_word_answer=bag_word_answer,
                               img_dir="./data_val/train.hdf5")
    val_dataset = VqaDataset(image_dir=test_image_dir,
                             question_json_file_path=test_question_path,
                             annotation_json_file_path=test_annotation_path,
                             image_filename_pattern="COCO_val2014_{}.jpg",
                             bag_word_question=bag_word_question,
                             bag_word_answer=bag_word_answer,
                             img_dir="./data_val/val.hdf5")

    num_question = train_dataset.bag_size_question
    num_answer = train_dataset.bag_size_answer

    # pdb.set_trace()

    max_len_train = train_dataset.max_len
    max_len_val = val_dataset.max_len

    print('max_len for train:{}, max_len for val:{}'.format(
        max_len_train, max_len_val))

    self._model = CoattentionNet(num_question, num_answer, 26)

    # self.optimizer = torch.optim.SGD(
    #     self._model.parameters(), lr=0.001, momentum=0.9)
    # pdb.set_trace()
    self.optimizer = torch.optim.Adam(
        self._model.parameters(), lr=4e-4, eps=1e-8)
    self.criterion = nn.CrossEntropyLoss().cuda()

    super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                     num_data_loader_workers=num_data_loader_workers)

  def get_bag_of_word_question(self):
    bag_word_question = {}
    max_len = 0
    for idx, entry in self.entries.items():

      question = entry['question']

      length = 0

      for word in question.lower().replace('?', '').split(' '):
        if word in bag_word_question:
          bag_word_question[word] += 1
        else:
          bag_word_question[word] = 1
        length += 1

      if max_len < length:
        max_len = length

    sorted_bag = sorted(bag_word_question.items(),
                        key=lambda bag: bag[1], reverse=True)
    bag_word_question_sorted = {key[0]: i for i,
                                key in enumerate(list(sorted_bag)[:2000])}

    return bag_word_question_sorted

  def get_bag_of_word_answer(self):
    bag_word_answer = {}

    # collect all the words
    for idx, entry in self.qa.items():
      answers = entry['answers']
      for answer in answers:
        if answer['answer'] in bag_word_answer:
          bag_word_answer[answer['answer']] += 1
        else:
          bag_word_answer[answer['answer']] = 1

    # sort the word bag and define the index
    sorted_bag = sorted(bag_word_answer.items(),
                        key=lambda bag: bag[1], reverse=True)
    bag_word_answer_sorted = {key[0]: i for i,
                              key in enumerate(list(sorted_bag)[:2000])}
    bag_word_answer_sorted.update({'other': 2000})

    return bag_word_answer_sorted

  def _optimize(self, predicted_answers, true_answer_ids):
    # TODO
    # raise NotImplementedError()
    loss = self.criterion(predicted_answers, true_answer_ids)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss
