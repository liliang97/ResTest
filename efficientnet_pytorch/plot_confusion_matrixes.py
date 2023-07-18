import sys
path_list=['/raid/home/zhangjianming3/lcq/workspace/LLib']
sys.path.append(path_list[0])
from model_att import EfficientNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score as accuracy_score_
import argparse
import sys
import os
import re
# path_list=['/raid/home/lcq/workspace/LLib']
# sys.path.append(path_list[0])
#from data.data import get_transform

pares=argparse.ArgumentParser(description='特征保存')
pares.add_argument('-P','--model-path',default='./weights/NWPU45_0.2_0.4_0.4_68_efficientnet-b0_att_labs0.1.pkl',type=str,help='要测试的模型的路径')
pares.add_argument('-G','--GPUs',type=str,default='[3]',help='GPU编号')
pares.add_argument('-S','--size',type=int,default=224,help='输入图片大小')
args=pares.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pare_dataset_name(path):
    path=path.strip('/')
    dataset_name=path.split('/')[-1]
    return dataset_name

def pare_num_class(path):
    dataset_name=pare_dataset_name(path)
    num_class=dataset_name.split('_')[-1]
    return int(num_class)


def createDataSet(train_image_root,train_transform,val_image_root,val_transform,test_image_root):
    train_dataset = ImageFolder(root=train_image_root, transform=train_transform)
    val_dataset = ImageFolder(root=val_image_root, transform=val_transform)
    test_dataset=ImageFolder(root=test_image_root, transform=val_transform)
    return train_dataset,val_dataset,test_dataset


def createDataLoader(train_dataset,val_dataset,test_dataset,train_batch_size,val_batch_size,test_batch_size):
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size)
    test_data_loder=DataLoader(dataset=test_dataset, batch_size=test_batch_size)
    return train_data_loader,val_data_loader,test_data_loder

def test_model(model,test_data_loader):
    model.eval()
    gt=[]
    pre=[]
    for data in test_data_loader:
        images, labels = data
        gt+=labels.tolist()
        images = images.to(device)
        att,features = model(images)
        _, pre_class = F.softmax(features, 1).max(1)
        pre+=pre_class.to(device).tolist()

    return gt,pre

data_set={
    '45':'NWPU-RESISC45',
    '30':'AID',
    '21':'UCMerced_LandUse',
    '31':'OPTIMAL'
}

def plot_confusion_matrix(y_true, y_pred, labels,dataset_name,save_path=None,show=True,normalized=True):
    assert save_path,'need save path'
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(15,15), dpi=120)

    # fig = plt.gcf()
    # fig.set_size_inches(7.0 / 3, 7.0 / 3)
    #plt.figure(dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag= 0 if normalized else 1
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if (intFlag):
            c = cm[y_val][x_val]
            if x_val == y_val:
                plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=18, va='center', ha='center',fontweight='medium')
            else:
                if c==0:
                    pass
                else:
                    plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=18, va='center', ha='center',fontweight='medium')

        else:
            c = cm_normalized[y_val][x_val]
            if (c >= 0.0045):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                #plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
                if x_val==y_val:
                    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
                else:
                    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                #plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
                pass
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap,aspect='auto')
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap,aspect='auto')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-',c='gray')
    #plt.gcf().subplots_adjust(bottom=0.15)
    if dataset_name=='UCMerced_LandUse':
        dataset_name='UCM'
    elif dataset_name=='NWPU-RESISC45':
        dataset_name='NWPU-RESISC'
    #plt.title('Confusion Matrix with {}({}% for train, {}% for test, OA:{:.2f}%)'.format(dataset_name,int(ratio*100),int(round(1-ratio,1)*100),acc*100),fontsize=12,fontweight='heavy')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=85)
    plt.yticks(xlocations, labels)
    # plt.xlabel('predict classes',font)
    # plt.ylabel('true classes',font)
    #plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8,hspace=0.2, wspace=0.3)
    #plt.margins(0, 0)
    plt.tight_layout()
    #fig.savefig(save_path, format='tif', transparent=True, dpi=300, pad_inches=0)
    plt.savefig(save_path, dpi=800)
    if show:
        plt.show()

#model save : ./weights/1_train_0.1_test_0.9_45_1575962216_resnet18_dan.pkl\nmodel save : ./weights/1_train_0.1_test_0.9_45_1575962216_resnet18_dan.pkl
def pare_model_path(txt_path):
    with open(txt_path,'r') as f:
        text=f.read()
    p=r'model save : (\./weights/[^\s]*\.pkl)'
    model_path=re.search(p,text).group(1)
    return model_path

def get_transform():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform

if __name__ == '__main__':
    assert args.model_path.endswith('pkl') or args.model_path.endswith('txt'),'未知的参数'
    if not os.path.exists('./matriex'):
        os.mkdir('./matriex')
    if args.model_path.endswith('pkl'):
        model_path=args.model_path
        info = args.model_path.split('/')[-1].split('_')[:5]
    else:
        model_path=pare_model_path(args.model_path)
        info = model_path.split('/')[-1].split('_')[:6]
    # test_image_root = '../dataset/{}/{}_{}_{}_{}_{}_{}/test'.format(data_set[info[-1]], *info)
    # train_image_root='../dataset/{}/{}_{}_{}_{}_{}_{}/train'.format(data_set[info[-1]], *info)
    # val_image_root='../dataset/{}/{}_{}_{}_{}_{}_{}/val'.format(data_set[info[-1]], *info)
    test_image_root = '/usr/LL/dataset/NWPU-RESISC45/NWPU45_0.2_0.4_0.4/test'
    train_image_root = '/usr/LL/dataset/NWPU-RESISC45/NWPU45_0.2_0.4_0.4/train'      #D:/dataset/UC21_0.8_0.1_0.1/train
    val_image_root = '/usr/LL/dataset/NWPU-RESISC45/NWPU45_0.2_0.4_0.4/val'
    save_path='./matriex/{}_{}_{}_{}_{}.png'.format(*info)
    labels=os.listdir(test_image_root)
    labels.sort()
    train_transform = get_transform()
    train_dataset, val_dataset, test_dataset = createDataSet(train_image_root, train_transform, val_image_root,
                                                             train_transform, test_image_root)
    _, _, test_data_loader = createDataLoader(train_dataset, val_dataset, test_dataset, 1, 1, 32)
    model = torch.load(model_path,map_location=torch.device('cuda'))
#    model = nn.DataParallel(model, eval(args.GPUs))
    model.to(device)
    gt, pre = test_model(model, test_data_loader)
    acc = accuracy_score_(gt, pre)
    print('test acc:{}'.format(acc))
    plot_confusion_matrix(gt,pre,labels,data_set['45'],save_path,show=None,normalized=True)
