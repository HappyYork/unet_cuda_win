import torch
import torch.nn.functional as F
import datetime
import torch.nn as nn
from my_dataset import VOCSegmentation
from train_and_eval import train_one_epoch,test_one_epoch
from torch.optim import lr_scheduler
import time
import os
import argparse
import transforms as T
class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self,channels):
        super(DownSample,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels,channels,3,2,1,bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)
class UpSample(nn.Module):
    def __init__(self,channels):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channels,channels//2,1,1)
    def forward(self,x,feature_map):
        up = F.interpolate(x,scale_factor=2,mode='nearest')
        out = self.layer(up)

        return torch.cat((out,feature_map),dim=1)

class Unet(nn.Module):
    def __init__(self,num_classes):
        super(Unet,self).__init__()
        self.num_classes = num_classes
        self.c1 = ConvBlock(3,64)
        self.d1 = DownSample(64)
        self.c2 = ConvBlock(64,128)
        self.d2 = DownSample(128)
        self.c3 = ConvBlock(128,256)
        self.d3 = DownSample(256)
        self.c4 = ConvBlock(256,512)
        self.d4 = DownSample(512)
        self.c5 = ConvBlock(512,1024)

        self.u1 = UpSample(1024)
        self.c6 = ConvBlock(1024,512)
        self.u2 = UpSample(512)
        self.c7 = ConvBlock(512,256)
        self.u3 = UpSample(256)
        self.c8 = ConvBlock(256,128)
        self.u4 = UpSample(128)
        self.c9 = ConvBlock(128,64)
        self.last = nn.Conv2d(64,num_classes,kernel_size=1)

    def forward(self,x):
        x1 = self.c1(x)         #torch.Size([., 64, 256, 256])
        x2 = self.d1(x1)        #torch.Size([., 64, 128, 128])
        x3 = self.c2(x2)        #torch.Size([., 128, 128, 128])
        x4 = self.d2(x3)        #torch.Size([., 128, 64, 64])
        x5 = self.c3(x4)        #torch.Size([., 256, 64, 64])
        x6 = self.d3(x5)        #torch.Size([.,256,32,32])
        x7 = self.c4(x6)        #torch.Size([.,512,32,32])
        x8 = self.d4(x7)        #torch.Size([., 512, 16, 16])
        x9 = self.c5(x8)        #torch.Size([., 1024, 16, 16])
        x10 = self.u1(x9,x7)    #torch.Size([., 1024, 32, 32])
        x11 = self.c6(x10)      #torch.Size([., 512, 32, 32])
        x12 = self.u2(x11,x5)   #torch.Size([., 1024, 64, 64])
        x13 = self.c7(x12)      #torch.Size([., 256, 64, 64])
        x14 = self.u3(x13,x3)   #torch.Size([., 256, 128, 128])
        x15 = self.c8(x14)      #torch.Size([., 128, 128, 128])
        x16 = self.u4(x15,x1)   #torch.Size([., 128, 256, 256])
        x17 = self.c9(x16)       #torch.Size([., 64, 256, 256])
        out = self.last(x17)
        return out

def parse_args():

    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="..\\..\\data\\", help="VOCdevkit root")           #modified the root path
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=15000, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--start-epoch', default= 12451, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--resume', default=True, help='resume from checkpoint')
    args = parser.parse_args()
    return args

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.2, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    #def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)
        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size,crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 256
    crop_size = 256
    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size,crop_size)

def create_model(num_classes):
    model = Unet(num_classes)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val_temp.txt")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    model = create_model(num_classes=num_classes)
    model.to(device)

    #optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)    #optimizer:2380

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,eps=1e-3)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=20,gamma = 0.01)

    if args.resume:
        path = 'save_weights/model_{}.pth'.format(args.start_epoch - 1 )
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    start_time = time.time()
    template = ("Epoch:{:2d},train_loss:{:.5f},train_acc:{:.2f}%")
    test_template = ("Epoch_t:{:2d},test_loss:{:.5f},test_acc:{:.2f}%")
    for epoch in range(args.start_epoch, args.epochs + 1):
        mean_loss,correct = train_one_epoch(model, optimizer, train_loader, device, epoch)
        # if exp_lr_scheduler:
        #     exp_lr_scheduler.step()
        print(template.format(epoch,mean_loss,correct*100))
        if (epoch % 50) == 0:
            test_loss,test_correct = test_one_epoch(model,optimizer,val_loader,device,epoch)

            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"train_correct:{correct:.4f}\n" \
                             f"test_correct:{test_correct:.4f}\n"
                f.write(train_info  + "\n\n")
            save_file = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args}
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
            print(test_template.format(epoch,test_loss,test_correct*100))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    main(args)
