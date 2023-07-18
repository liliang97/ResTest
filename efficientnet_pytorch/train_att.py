import  torch
import  argparse
import  os
import  time
import  torch.utils.model_zoo as model_zoo
import  torch.nn.functional as F

from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torchvision.datasets import ImageFolder
from    torch import nn, optim
from    torch.optim.lr_scheduler import CosineAnnealingLR,StepLR

#from    model import EfficientNet
#from    efficientNet import EfficientNet
from    model_att import EfficientNet
#from    model_SA_fc import EfficientNet

from    labelSmooth import LabelSmoothCELoss

from utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

parser = argparse.ArgumentParser(description='Remote sensing image classification')
# parser.add_argument('-D','--image_root', default = '/usr/LL/dataset/AID_0.5_0.25_0.25', type=str, help='the root path of train data')
parser.add_argument('-D','--image_root', default = 'D:/文档/pytorch/AID/AID_dataset/AID_0.5_0.25_0.25', type=str, help='the root path of train data')
#parser.add_argument('-D','--image_root', default = '/usr/LL/dataset/SIRI-WHU_0.5_0.25_0.25', type=str, help='the root path of train data')
parser.add_argument('--epoches', default=120, type=int, help='number of total epochs to run')
parser.add_argument('-B','--train_batch', default=16, type=int, help='train batchsize (default: 64)')
parser.add_argument('-b','--val_batch', default=16, type=int, help='val batchs ize (default: 64)')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--weight_root', default='./weights', type=str, help='the path of weights will save')
parser.add_argument('--log_root', default='./log', type=str, help='the path of log will save')
parser.add_argument('--val_every', default=2, type=int, help='val every ')
parser.add_argument('--num_classes', default=30, type=int, help='number of classes')
parser.add_argument('--useDataEnhancer', default='True', type=str, help='if use Data Enhancer')
parser.add_argument('--model_name', default='efficientnet-b0_attloss', type=str, help='the name of model')
parser.add_argument('--weight_cent', default=0.008, type=float, help='weight_cent')
parser.add_argument('--change', default=30, type=int, help='change')
parser.add_argument('--times', default=0.1, type=float, help='times')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    image_root = args.image_root
    epoches = args.epoches
    train_batch = args.train_batch
    val_batch = args.val_batch
    lr = args.lr
    weight_root = args.weight_root
    log_root = args.log_root
    weight_cent = args.weight_cent
    val_every = args.val_every
    model_name = args.model_name
    num_classes = args.num_classes
    GPUs = True

    # 把目录和文件名合成一个路径
    train_image_root = os.path.join(image_root, 'train')
    val_image_root = os.path.join(image_root, 'val')
    test_image_root = os.path.join(image_root, 'test')

    train_transform = transforms.Compose([
        transforms.Resize((224,  224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(root=train_image_root, transform=train_transform)
    val_dataset = ImageFolder(root=val_image_root, transform=train_transform)
    test_dataset = ImageFolder(root=test_image_root, transform=train_transform)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True, num_workers=0)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=val_batch, num_workers=0)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=val_batch, num_workers=0)

    dataset_name = pare_dataset_name(image_root)

    model_name_self = model_name[0:15]
    model = EfficientNet.from_pretrained(model_name_self, num_classes=num_classes).to(device)
    load_pretrained_weights(model, model_name_self, weights_path=None, advprop=False)
    #model.load_state_dict(torch.load("./weights/AID_0.8_0.1_0.1_0_efficientnet-b0.pkl")).to(device)
    # model = torch.load("./weights/AID_0.8_0.1_0.1_1_efficientnet-b0.pkl").to(device)

    #criterion = nn.CrossEntropyLoss().to(device)
    criterion = LabelSmoothCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model)

    # 学习率调整
    lr_scheduler = StepLR(optimizer, args.change, args.times)
    iter_number = 0
    val_acc = 0.0
    max_val_epoch = 0
    date = int(time.time())

    log_name = '{}_{}_{}_{}.txt'.format('log', model_name, dataset_name, date)
    log_root_name = os.path.join(log_root, log_name)


    for epoch in range(epoches):

        model, val_acc, iter_number = train_one_epoch(model, optimizer, criterion, epoch, epoches, iter_number, log_root_name,
                                                      train_data_loader, iter_number_print=10, val_acc=val_acc,
                                                      max_val_epoch=max_val_epoch, weight_cent=weight_cent)
        lr_scheduler.step()
        # if epoch % val_every == 0:  # val_every = 2
        #     val_acc, max_val_epoch = val_one_epoch(model, model_name, val_data_loader, epoch, weight_root,
        #                                            dataset_name, log_root_name,
        #                                            val_acc=val_acc, max_val_epoch=max_val_epoch)
        val_acc, max_val_epoch = val_one_epoch(model, model_name, val_data_loader, epoch, weight_root,
                                               dataset_name, log_root_name,
                                               val_acc=val_acc, max_val_epoch=max_val_epoch)

    ###test####
    del model
    model_name = '{}_{}_{}.pkl'.format(dataset_name, max_val_epoch, model_name)
    save_path = os.path.join(weight_root, model_name)
    model = torch.load(save_path)
    model.to(device)
    test_acc = test_model(model, test_data_loader)
    with open(log_root_name, "a") as f:
        f.write('test acc:{}'.format(test_acc) + '\n')
    print(epoch, 'test acc:', test_acc)


def train_one_epoch(model, optimizer, criterion, epoch, epoches, iter_number, log_root_name,
                    train_data_loader,
                    iter_number_print=10, val_acc=0, max_val_epoch=0, weight_cent=0.006):
    model.train()
    criterion.train()
    # data一次为batch size(128)张图片，共有(AID图片总数)9000/128=70次循环
    for data in train_data_loader:
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        att_outputs, per_outputs = model(images)
        att_loss = criterion(att_outputs, labels)
        per_loss = criterion(per_outputs, labels)
        loss = att_loss + per_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for param in criterion.parameters():
        #     param.grad.data *= (1. / weight_cent)
        # optimizer.step()

        # 每10次输出一次loss和acc
        if iter_number % iter_number_print == 0:
            batch_size = images.size(0)
            _, pre_class = F.softmax(per_outputs, 1).max(1)
            right = (pre_class == labels).sum().item()
            log = '[{}/{} {}th]||loss:{:.7f}||acc:{:.7f}||max val acc:{:.7}[{}]'.format(
                epoch, epoches, iter_number, loss.data.item(),
                right / batch_size, val_acc, max_val_epoch)
            with open(log_root_name, "a") as f:
                f.write(log + '\n')
            print(log)
        iter_number += 1
        del images
        del labels
    return model, val_acc, iter_number


def val_one_epoch(model,model_name, val_data_loader, current_epoch, weight_root, datasetName, log_root_name, val_acc=0,
                  max_val_epoch=0):
    model.eval()
    right = 0
    batch_size = 0
    for data in val_data_loader:
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        att_outputs, per_outputs = model(images)
        # 为了确保各个预测结果的概率之和等于1。我们只需要将转换后的结果进行归一化处理。
        # 方法就是将转化后的结果除以所有转化后结果之和，可以理解为转化后结果占总数的百分比。
        # softmax函数：1）分子：通过指数函数，将实数输出映射到零到正无穷。2）分母：将所有结果相加，进行归一化。
        # max(1)： max函数得到准确率最高的类别下标
        _, pre_class = F.softmax(per_outputs, 1).max(1)
        # 如果pre_class等于已知的下标值，则判断正确的图片数加1
        right += (pre_class == labels).sum().item()
        batch_size += images.size(0)

        del images
        del labels
    acc = right / batch_size

    log = 'val acc:{:.7f}'.format(acc)
    with open(log_root_name, "a") as f:
        f.write(log + '\n')
    print(log)


    if acc >= val_acc:
        val_acc = acc
        max_val_epoch = current_epoch
        # 记录
        save_name = '{}_{}_{}.pkl'.format(datasetName, max_val_epoch, model_name)
        save_path = os.path.join(weight_root, save_name)
        torch.save(model, save_path)
        model_log = 'model save : {}'.format(save_path)
        with open(log_root_name, "a") as f:
            f.write(model_log + '\n')

    return val_acc, max_val_epoch


def test_model(model, test_data_loader):
    model.eval()
    right = 0
    batch_size = 0
    for data in test_data_loader:
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        att_outputs, per_outputs = model(images)
        _, pre_class = F.softmax(per_outputs, 1).max(1)
        right += (pre_class == labels).sum().item()
        batch_size += images.size(0)

    acc = right / batch_size
    return acc


def pare_dataset_name(path):
    path = path.strip('/')
    dataset_name = path.split('/')[-1]
    return dataset_name


if __name__ == '__main__':
    main()
