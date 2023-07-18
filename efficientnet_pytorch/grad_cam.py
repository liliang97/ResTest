import torch
import os
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import re
data_set={
    '45':'NWPU-RESISC45',
    '30':'AID',
    '21':'UCMerced_LandUse',
    '31':'OPTIMAL'
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=int,default=-1,
                        help='Use NVIDIA GPU acceleration')

    parser.add_argument('--model_path', type=str, default=r'./weights/UC21_0.8_0.1_0.1_59_efficientnet-b0_att_labs0.1.pkl',
                        help='Input image model')

    parser.add_argument('--size', type=int, default=224,
                        help='Input image size')

    args = parser.parse_args()
    return args

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model):
        self.model = model
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        #[1,1280,1,1]
        a,y = self.model.extract_features_att(x)
        y.register_hook(self.save_gradient)
        outputs += [y]
        y = self.model._avg_pooling(y)
        y = y.flatten(start_dim=1)
        y = self.model._dropout(y)
        y = self.model._fc(y)

        return outputs, y



def create_heatmap(model_path,image_path,GPU_ID):
    model = torch.load(model_path,map_location=torch.device('cpu'))
    grad_cam = GradCam(model, GPU_ID=GPU_ID)
    # imread(image_path, 1) 读入一副彩色图片，忽略alpha通道，可用1作为实参替代.
    # alpha通道，又称A通道，是一个8位的灰度通道，该通道用256级灰度来记录图像中的透明度复信息，定义透明、不透明和半透明区域，其中黑表示全透明，白表示不透明，灰表示半透明
    img = cv2.imread(image_path, 1)
    img = np.float32(cv2.resize(img, (args.size, args.size))) / 255
    input = preprocess_image(img)
    target_index = None
    mask = grad_cam(input, target_index)
    return img,mask

class GradCam:
    def __init__(self, model, GPU_ID):
        self.model = model
        self.model.eval()
        self.GPU_ID = GPU_ID
        if self.GPU_ID>=0:
            self.model = model.cuda(GPU_ID)

        self.extractor = FeatureExtractor(self.model)

    def get_gradients(self):
        return self.extractor.gradients

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.GPU_ID>=0:
            features, output = self.extractor(input.cuda(self.GPU_ID))
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.GPU_ID>=0:
            one_hot = torch.sum(one_hot.cuda(self.GPU_ID) * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam



def save_cam_on_image(img, mask, save_path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_path,cv2.resize(np.uint8(255 * cam),(320,320)))


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input

def pare_model_path(txt_path):
    with open(txt_path,'r') as f:
        text=f.read()
    p=r'model save : (\./weights/[^\s]*\.pkl)'
    model_path=re.search(p,text).group(1)
    return model_path


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    assert args.model_path.endswith('pkl') or args.model_path.endswith('txt'),'未知的参数'
    if args.model_path.endswith('pkl'):
        model_path=args.model_path
        info = args.model_path.split('/')[-1].split('_')[:5]
    else:
        model_path=pare_model_path(args.model_path)

    weight_name=model_path.split('/')[-1]
    temp=weight_name.split('_')[0:4]
    #dir_name:生成根目录
    dir_name='{}_{}_{}_{}_{}'.format('efficientNet_att_labs',*temp)
    #路径
    dataset_name=data_set['21']
    root_dir='D:/dataset/UC0.8/test'
    dst_dir='./heatmap/{}/{}/test'.format(dataset_name,dir_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    sub_dir_names=os.listdir(root_dir)
    for sub_dir_name in sub_dir_names:
        image_names=os.listdir(os.path.join(root_dir,sub_dir_name))
        if not os.path.exists(os.path.join(dst_dir,sub_dir_name)):
            os.mkdir(os.path.join(dst_dir,sub_dir_name))
        for image_name in image_names:
            image_path=os.path.join(root_dir,sub_dir_name,image_name)
            img, mask=create_heatmap(model_path,image_path,args.use_cuda)
            save_path=os.path.join(dst_dir,sub_dir_name,image_name)
            save_cam_on_image(img, mask,save_path)
            print(image_name)
