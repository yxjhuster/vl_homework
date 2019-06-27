import sys
sys.path.append('../')
from torch.utils.data import Dataset
from external.vqa.vqa import VQA
import numpy as np
import torch
import h5py
import pdb


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern, bag_word_question, bag_word_answer, img_dir):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """

        # could define the max_len from outside
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

        self.milky_vector_question_coattention, self.max_len = self.get_milky_vector_question_coattention()
        self.milky_vector_answer = self.get_milky_vector_answer()

        self.max_len = 26

        self.mask_dict = self.get_mask()

        self.gt_dict = self.get_gt()
        self.question_index = self.get_index()

        self.f = h5py.File(img_dir, "r")
        # self.img_dir = img_dir

        # pdb.set_trace()

    # get the question index list
    def get_index(self):
        index_list = []
        for idx, entry in self.qa.items():
            index_list.append(idx)
        return index_list

    # get the index of ground truth
    def get_gt(self):
        gt_dict = {}
        bag_list = list(self.bag_word_answer.keys())
        for idx, entry in self.qa.items():
            gt = entry['multiple_choice_answer']
            for i, key in enumerate(bag_list):
                if(key == gt):
                    break
            gt_dict[idx] = i
        return gt_dict

    # get milky vector of the question
    def get_milky_vector_question_coattention(self):
        milky_vector_question = {}
        max_len = 0
        for idx, entry in self.entries.items():
            question = entry['question']
            bag = []
            for word in question.lower().replace('?', '').split(' '):
                if word in self.bag_word_question:
                    # vector = np.zeros(self.bag_size_question)
                    # vector[self.bag_word_question[word]] = 1
                    bag.append(self.bag_word_question[word])
                    if(len(bag) > max_len):
                        max_len = len(bag)
            milky_vector_question[idx] = np.array(bag)

        return milky_vector_question, max_len

    # get mask of the questions
    def get_mask(self):
        mask_dict = {}
        for idx, entry in self.milky_vector_question_coattention.items():
            mask = np.concatenate(
                (np.ones(len(entry)), np.zeros(self.max_len - len(entry))))
            mask_dict.update({idx: mask})

        return mask_dict

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
        img_idx = self.entries[idx]['image_id']
        # with h5py.File(self.img_dir, 'r') as f:
        #     feature = f[str(img_idx)][0]
        feature = self.f[str(img_idx)][:]
        feature = torch.tensor(feature, dtype=torch.float32)
        # question_vector = torch.tensor(
        #     self.milky_vector_question_coattention[idx], dtype=torch.float32)
        question_vector = self.milky_vector_question_coattention[idx]
        gt = torch.tensor(self.gt_dict[idx], dtype=torch.long)
        mask = torch.tensor(self.mask_dict[idx], dtype=torch.float32)
        item.update({'feature': feature})
        item.update({'question': question_vector})
        item.update({'gt': gt})
        item.update({'mask': mask})
        return item
