import sys
sys.path.append('../')
from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
from external.vqa.vqa import VQA
import torch
import torch.nn as nn
import copy
import pdb


class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
  """
  Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
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

    pdb.set_trace()

    train_dataset = VqaDataset(image_dir=train_image_dir,
                               question_json_file_path=train_question_path,
                               annotation_json_file_path=train_annotation_path,
                               image_filename_pattern="COCO_train2014_{}.jpg",
                               bag_word_question=bag_word_question,
                               bag_word_answer=bag_word_answer)

    val_dataset = VqaDataset(image_dir=test_image_dir,
                             question_json_file_path=test_question_path,
                             annotation_json_file_path=test_annotation_path,
                             image_filename_pattern="COCO_val2014_{}.jpg",
                             bag_word_question=bag_word_question,
                             bag_word_answer=bag_word_answer)

    # num_question = train_dataset.bag_size_question
    # num_answer = train_dataset.bag_size_answer
    model = SimpleBaselineNet(2000, 2001)

    self.optimizer = torch.optim.SGD(
        [{'params': model.fc.parameters()},
         {'params': model.feature.parameters()},
         {'params': model.embedding.parameters(), 'lr': 0.8}], lr=0.01, momentum=0.9)

    self.criterion = nn.CrossEntropyLoss().cuda()

    super().__init__(train_dataset, val_dataset, model,
                     batch_size, num_epochs, num_data_loader_workers)

  def get_bag_of_word_question(self):
    bag_word_question = {}
    for idx, entry in self.entries.items():

      question = entry['question']

      for word in question.lower().replace('?', '').split(' '):
        if word in bag_word_question:
          bag_word_question[word] += 1
        else:
          bag_word_question[word] = 1

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
    loss = self.criterion(predicted_answers, true_answer_ids)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss
