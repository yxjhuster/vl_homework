import sys
sys.path.append('../')
from torch.utils.data import Dataset
from external.vqa.vqa import VQA
import os
import cv2
import numpy as np
import torch
# import torch.nn as nn
import pdb


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern, bag_word_question, bag_word_answer):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self.image_dir = image_dir
        self.image_filename_pattern = image_filename_pattern
        self.vqa_loader = VQA(annotation_file=annotation_json_file_path,
                              question_file=question_json_file_path)

        self.entries = self.vqa_loader.qqa
        self.qa = self.vqa_loader.qa

        self.bag_word_question = bag_word_question
        self.bag_word_answer = bag_word_answer

        self.bag_size_question = len(self.bag_word_question)
        self.bag_size_answer = len(self.bag_word_answer)

        self.milky_vector_question = self.get_milky_vector_question()
        self.milky_vector_answer = self.get_milky_vector_answer()

        self.gt_dict = self.get_gt()
        self.question_index = self.get_index()

        # if 'val' in image_dir:
        #     pdb.set_trace()

        # pdb.set_trace()

    # get the question index list
    def get_index(self):
        index_list = []
        for idx, entry in self.qa.items():
            index_list.append(idx)
        return index_list

    # # get the index of ground truth
    # def get_gt(self):
    #     gt_dict = {}
    #     bag_list = list(self.bag_word_answer.keys())
    #     for idx, entry in self.qa.items():
    #         gt = entry['multiple_choice_answer']
    #         for i, key in enumerate(bag_list):
    #             if(key == gt):
    #                 break
    #         gt_dict[idx] = i
    #     return gt_dict

    # get the index of ground truth
    def get_gt(self):
        gt_dict = {}
        # bag_list = list(self.bag_word_answer.keys())
        for idx, entry in self.qa.items():
            gt = entry['multiple_choice_answer']
            # for i, key in enumerate(bag_list):
            #     if(key == gt):
            #         break
            # gt_dict[idx] = i
            if gt in self.bag_word_answer:
                gt_dict.update({idx: self.bag_word_answer[gt]})
            else:
                gt_dict.update({idx: self.bag_word_answer['other']})
        return gt_dict

    # get milky vector of the question
    def get_milky_vector_question(self):
        milky_vector_question = {}
        for idx, entry in self.entries.items():
            question = entry['question']
            bag = []
            for word in question.lower().replace('?', '').split(' '):
                if word in self.bag_word_question:
                    bag.append(self.bag_word_question[word])
            one_hot = np.zeros(self.bag_size_question)
            one_hot[bag] = 1
            milky_vector_question[idx] = np.array(one_hot)

        return milky_vector_question

    # get milky vector of the answer
    def get_milky_vector_answer(self):
        milky_vector_answer = {}
        for idx, entry in self.qa.items():
            answers = entry['answers']
            bag = []
            for answer in answers:
                if answer['answer'] in self.bag_word_answer:
                    bag.append(self.bag_word_answer[answer['answer']])
                else:
                    bag.append(self.bag_word_answer['other'])
            milky_vector_answer[idx] = np.array(bag)

        return milky_vector_answer

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        idx = self.question_index[idx]
        item = {}
        image_id = self.entries[idx]['image_id']
        image_id = '{0:012d}'.format(image_id)
        path = os.path.join(
            self.image_dir, self.image_filename_pattern.format(image_id))
        feature = cv2.imread(path)
        feature = cv2.resize(feature, (248, 248))
        feature = np.array(feature.astype(np.float32))
        feature = torch.tensor(feature, dtype=torch.float32)
        question_vector = torch.tensor(
            self.milky_vector_question[idx], dtype=torch.float32)
        gt = torch.tensor(self.gt_dict[idx], dtype=torch.long)
        item.update({'feature': feature})
        item.update({'question': question_vector})
        item.update({'gt': gt})
        return item
