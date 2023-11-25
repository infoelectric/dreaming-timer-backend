from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import cv2
import dlib
import numpy as np
# from model import Net
import torch
from imutils import face_utils

class eyes_dataset(Dataset):
    def __init__(self, x_file_paths, y_file_path, transform=None):
        self.x_files = x_file_paths
        self.y_files = y_file_path
        self.transform = transform


    def __getitem__(self, idx):
        x = self.x_files[idx]
        x = torch.from_numpy(x).float()

        y = self.y_files[idx]
        y = torch.from_numpy(y).float()

        return x, y

    def __len__(self):
        return len(self.x_files)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.reshape(-1, 1536)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to('cpu')
summary(model, (1,26,34))
IMG_SIZE = (34,26)
PATH = './minsun/path/path.pth'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./python/shape_predictor_68_face_landmarks.dat')

model = Net()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

def predict(pred):
  pred = pred.transpose(1, 3).transpose(2, 3)

  outputs = model(pred)

  pred_tag = torch.round(torch.sigmoid(outputs))

  return pred_tag

def letsgo(img_path):
  img_ori = cv2.imread(img_path)
  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)
  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])


    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    # cv2.imshow('l', eye_img_l)
    # cv2.imshow('r', eye_img_r)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)


    eye_input_l = torch.from_numpy(eye_input_l)
    eye_input_r = torch.from_numpy(eye_input_r)


    pred_l = predict(eye_input_l)
    pred_r = predict(eye_input_r)

    if pred_l.item() == 0.0 and pred_r.item() == 0.0:
      return 0;

    else:
      return 1;


