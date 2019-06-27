from resnet import resnet18
import sys
sys.path.append('../')
from external.vqa.vqa import VQA
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import pdb

train_question_path = '../data/OpenEnded_mscoco_val2014_questions.json'
train_annotation_path = '../data/mscoco_val2014_annotations.json'
image_filename_pattern = "COCO_val2014_{}.jpg"
image_dir = '../data/val2014/'

# vqa_loader = VQA(annotation_file=train_annotation_path,
#                  question_file=train_question_path)


class imgDataset(Dataset):
    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern):
        self.image_dir = image_dir
        self.image_filename_pattern = image_filename_pattern
        self.vqa_loader = VQA(annotation_file=train_annotation_path,
                              question_file=train_question_path)
        self.img_id = self.vqa_loader.getImgIds()
        self.img_id = list(dict.fromkeys(self.img_id))

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        idx = self.img_id[idx]
        i = '{0:012d}'.format(idx)
        item = {}
        path = os.path.join(
            self.image_dir, self.image_filename_pattern.format(i))
        feature = cv2.imread(path)
        feature = cv2.resize(feature, (448, 448))
        feature = np.array(feature.astype(np.float32) / 255)
        feature = np.transpose(feature, (2, 0, 1))
        feature = torch.tensor(
            feature, dtype=torch.float32, requires_grad=False)
        item.update({'idx': idx})
        item.update({'feature': feature})
        return item


dataset = imgDataset(image_dir, train_question_path,
                     train_annotation_path, image_filename_pattern)
batch_size = 200

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)

img_feat = {}

f = h5py.File("./data_val/val.hdf5", "w")

# pdb.set_trace()

model = resnet18(pretrained=True).cuda()

model.eval()

j = 0

for batch_id, batch_data in enumerate(dataloader):
    # pdb.set_trace()
    feature = batch_data['feature'].cuda()
    with torch.no_grad():
        output = model(feature)
    idx = batch_data['idx'].numpy()
    output_numpy = output.data.cpu().numpy()
    shape = output_numpy.shape
    # pdb.set_trace()
    for i in range(len(output_numpy)):
        ds = f.create_dataset(str(idx[i]), data=output_numpy[i], dtype='f')
        # img_feat.update({: output_numpy[i]})

    # pdb.set_trace()

    print('iteration {}'.format(batch_id))

# with open('./data/image_train.pkl', 'wb') as f:
#     pickle.dump(img_feat, f, pickle.HIGHEST_PROTOCOL)
# pdb.set_trace()

# length = len(img_id)

# for idx in img_id:
#     i = '{0:012d}'.format(idx)
#     path = os.path.join(
#         image_dir, image_filename_pattern.format(i))

#     feature = cv2.imread(path)
#     # pdb.set_trace()
#     feature = cv2.resize(feature, (500, 500))
#     feature = np.array(feature.astype(np.float32) / 255)
#     feature = np.transpose(feature, (2, 1, 0))
#     feature = torch.tensor([feature], dtype=torch.float32).cuda()

#     output = model(feature)
#     output_numpy = output.data.cpu().numpy()

#     # pdb.set_trace()

#     img_feat.update({idx: np.squeeze(output_numpy.T)})
#     j += 1

#     if j % 100 == 0:
#         print('iteration {}/{}'.format(j, length))
# pdb.set_trace()

# with open('data.json', 'w') as fp:
#     json.dump(img_feat, fp)
