import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import torch

from    model_att import EfficientNet
#from  model import EfficientNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# input image
LABELS_file = 'imagenet-simple-labels.json'
image_file = 'D:/dataset/AID0.2/test/Airport/airport_2.jpg'

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape  #1,320,7,7
    output_cam = []
    for idx in class_idx:       #weight_softmax:30,1280
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

if __name__ == '__main__':


    net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=30).to(device)
    finalconv_name = '_blocks_last'
    net.eval()

    # hook the feature extractor
    features_blobs = []

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-5].data.numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # load test image
    img_pil = Image.open(image_file)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    att,logit = net(img_variable)

    # load the imagenet category list
    # with open(LABELS_file) as f:
    #     classes = json.load(f)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    # for i in range(0, 5):
    #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
#    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
    img = cv2.imread(image_file)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('./visualize/cam/CAM.jpg', result)
