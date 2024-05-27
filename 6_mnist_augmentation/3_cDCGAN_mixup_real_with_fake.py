'''

Step 3
cDCGAN AS Surrogate Model
Mixup Real Images with Fake Images

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import torchvision
import os
from PIL import Image
from pytorch_fid import fid_score
import numpy as np
import time
from shutil import copyfile
from pytorch_fid import fid_score
import matplotlib.pyplot as plt
import random
import re

batch_size = 64
train_epoch = 1500
check_epoch = 15
picture_num_generate = 600
picture_num_calculate = 600
z_dimension = 100  # noise dimension
dataset_path = './epoch_460/'
alpha = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ' + device)

def onehot(x, class_count=10):
    return torch.eye(class_count)[x, :]

if not os.path.isdir('MNIST_cDCGAN_mixup_between'):
    os.mkdir('MNIST_cDCGAN_mixup_between')
if not os.path.isdir('MNIST_cDCGAN_mixup_between/Fixed_results'):
    os.mkdir('MNIST_cDCGAN_mixup_between/Fixed_results')
if not os.path.isdir('MNIST_cDCGAN_mixup_between/Backup'):
    os.mkdir('MNIST_cDCGAN_mixup_between/Backup')
if not os.path.isdir('dc_img_mixup_between'):
    os.mkdir('dc_img_mixup_between')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
fid_hist = {}
fid_hist['epoch_num'] = []
fid_hist['FID_score_mnist'] = []
fid_hist['FID_score_victim'] = []

fixed_z_ = torch.randn(batch_size, 100).to(device)
fixed_y_ = torch.randint(0, 10,size=(batch_size,))
fixed_y_label_ = onehot(fixed_y_).to(device)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

#Define loading of self-built dataset based on pathes saved in txt file
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        super(MyDataset,self).__init__()
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs 
            self.transform = transform
            self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert("L")
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    def __len__(self):
        return len(self.imgs)

dataset = MyDataset(txt_path = "./Dataset.txt", transform=img_transform)
print("Totle number of data loaded: " + str(len(dataset)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 

#Define generator with 100-length noise and 10-length target as inputs, [batchsize, 1, 28, 28] image tensor as output
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc = nn.Linear(110, 784)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )
 
    def forward(self, x, tar):
        input = torch.cat([x, tar], 1)
        input = self.fc(input)
        input = input.view(input.size(0), 1, 28, 28)
        input = self.br(input)
        input = self.downsample1(input)
        input = self.downsample2(input)
        input = self.downsample3(input)
        return input
    
#Define discriminator with the image tensor and 10-length target as input and a binary classification as output
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.linear = nn.Sequential(nn.Linear(10, 1 * 28 * 28),nn.Sigmoid())
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.cat = nn.Linear(2*28*28,1*28*28)
 
    def forward(self, x, tar):
        tar = self.linear(tar)
        x = torch.reshape(x,(-1, 1*28*28))
        input = torch.cat([x, tar], 1)
        input = self.cat(input)
        input = torch.reshape(input,(-1, 1, 28, 28))
        input = self.conv1(input)
        input = self.conv2(input)
        input = input.view(input.size(0), -1)
        input = self.fc(input)
        return input

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_fid_mnist_hist(hist, show = False, save = False, path = 'FID_hist.png'):
    x = hist['epoch_num']
    y = hist['FID_score_mnist']

    plt.plot(x, y)

    plt.xlabel('Epoch')
    plt.ylabel('FID')

    plt.title('FID score to MNIST dataset')
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_fid_victim_hist(hist, show = False, save = False, path = 'FID_hist.png'):
    x = hist['epoch_num']
    y = hist['FID_score_victim']

    plt.plot(x, y)

    plt.xlabel('Epoch')
    plt.ylabel('FID')

    plt.title('FID score to victim model generated images')
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

#Concatenate each batch of real images with a corresponding batch of fake images and then performing shuffling, do the same to labels and targets
def one_batch(real,fake,target):
    data = torch.cat((real, fake))
    ones = torch.ones(real.size(0), 1)
    zeros = torch.zeros(fake.size(0), 1)
    labels = torch.cat((ones, zeros))
    targets = torch.cat((target, target))
    batch_size = data.size()[0]
    index = torch.randperm(batch_size)
    return data[index], labels[index], targets[index]

D = discriminator().to(device)
G = generator().to(device)

num_total_learnable_params_G = sum(i.numel() for i in G.parameters() if i.requires_grad)
print('Number of learnable params of Generator: {}.'.format(num_total_learnable_params_G))      #G: 99177
num_total_learnable_params_D = sum(i.numel() for i in D.parameters() if i.requires_grad)
print('Number of learnable params of Discriminator: {}.'.format(num_total_learnable_params_D))  #D: 4504129
 
criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.00002, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G.train()#Enter training mode
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for i, (img, target) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        num_img = img.size(0)
        real_img = img
        target = F.one_hot(target, num_classes = 10)
        target = target.to(torch.float32) #To fix RuntimeError: mat1 and mat2 must have the same dtype

        #Train discriminator with mixup enhancement
        z = torch.randn(num_img, z_dimension, device=device)
        fake_img = G(z,target)

        d1, l1, t1 = one_batch(real_img,fake_img,target)
        d2, l2, t2 = one_batch(real_img,fake_img,target)

        #Perform the mixup on two batches of data got above
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * d1 + (1 - lam) * d2
        mixed_label = lam * l1 + (1 - lam) * l2
        mixed_target = lam * t1 + (1 - lam) * t2
        mixed_out = D(mixed_img.detach(),mixed_target.detach())
        d_loss = criterion(mixed_out, mixed_label.to(device))

        d_optimizer.zero_grad()
        d_loss.backward()
        if d_loss.item() > 0.1:
            d_optimizer.step()
        D_losses.append(d_loss.item())
 
        #Train generator
        output = D(fake_img,target)
        real_label = torch.ones_like(output)
        g_loss_fake = criterion(output, real_label)

        g_optimizer.zero_grad()
        g_loss_fake.backward()
        g_optimizer.step()
        G_losses.append(g_loss_fake.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    
    G.eval()
    with torch.no_grad():
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
        
        fixed_p = 'MNIST_cDCGAN_mixup_between/Fixed_results/MNIST_cDCGAN_' + str(epoch + 1) + '.png'
        test_images = G(fixed_z_, fixed_y_label_)
        test_images = (test_images+1)/2
        save_image(test_images.clamp(0,1), fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        #Save output images every few epochs
        if (epoch+1)%check_epoch == 0:
            torch.save(G, 'MNIST_cDCGAN_mixup_between/Backup/generator_epoch_{}.pth'.format(epoch+1))
            torch.save(D, 'MNIST_cDCGAN_mixup_between/Backup/discriminator_epoch_{}.pth'.format(epoch+1))
            if not os.path.exists('./dc_img_mixup_between/epoch_'+str(epoch+1)):
                os.mkdir('./dc_img_mixup_between/epoch_'+str(epoch+1))
            for label in range(10):
                label_list = []
                label_list.append(label)
                label_input = torch.tensor(label_list)
                label_input_onehot=torch.eye(10)[label_input, :]

                for g in range(picture_num_generate):
                    if not os.path.exists('./dc_img_mixup_between/epoch_'+str(epoch+1)+'/'+str(label)):
                        os.mkdir('./dc_img_mixup_between/epoch_'+str(epoch+1)+'/'+str(label))
                    if not os.path.exists('./dc_img_mixup_between/epoch_'+str(epoch+1)+'/g_all'):
                        os.mkdir('./dc_img_mixup_between/epoch_'+str(epoch+1)+'/g_all')
                    test_input = torch.randn(1, 100, device = device)
                    prediction = G(test_input, label_input_onehot.to(device))
                    prediction_img = (prediction+1)/2
                    save_image(prediction_img.clamp(0,1), './dc_img_mixup_between/epoch_{}/{}/fake_images_{}_{}.png'.format(epoch+1,label,label,g+1), nrow=1, format='png')
                    if g < picture_num_calculate:
                        copyfile('./dc_img_mixup_between/epoch_{}/{}/fake_images_{}_{}.png'.format(epoch+1,label,label,g+1), './dc_img_mixup_between/epoch_{}/g_all/fake_images_{}_{}.png'.format(epoch+1,label,g+1))
            print("Image of epoch "+str(epoch+1)+" saved.")

            #Calculate fid scores to MNIST and to victim output and save in dictionary
            fid_mnist = fid_score.calculate_fid_given_paths(['./dc_img_mixup_between/epoch_{}/g_all'.format(epoch+1),
                                                            '../o_img_6000'],
                                                            batch_size=1,
                                                            dims = 2048,
                                                            device = device,
                                                            num_workers=1
                                                            )
            fid_victim = fid_score.calculate_fid_given_paths(['./dc_img_mixup_between/epoch_{}/g_all'.format(epoch+1),
                                                            dataset_path+'g_all'],
                                                            batch_size=1,
                                                            dims = 2048,
                                                            device = device,
                                                            num_workers=1
                                                            )
            print('FID: '+str(fid_mnist))
            fid_hist['epoch_num'].append(epoch+1)
            fid_hist['FID_score_mnist'].append(fid_mnist)
            fid_hist['FID_score_victim'].append(fid_victim)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G, "MNIST_cDCGAN_mixup_between/generator_param.pth")
torch.save(D, "MNIST_cDCGAN_mixup_between/discriminator_param.pth")

show_train_hist(train_hist, save=True, path='MNIST_cDCGAN_mixup_between/MNIST_cDCGAN_train_hist.png')
show_fid_mnist_hist(fid_hist, save=True, path='MNIST_cDCGAN_mixup_between/MNIST_cDCGAN_FID_hist_mnist.png')
show_fid_victim_hist(fid_hist, save=True, path='MNIST_cDCGAN_mixup_between/MNIST_cDCGAN_FID_hist_victim.png')

fid_log = "Epoch = " + str(fid_hist['epoch_num']) + "\n\nMNIST = " + str(fid_hist['FID_score_mnist']) + "\n\nVictim = " + str(fid_hist['FID_score_victim'])
with open('MNIST_cDCGAN_mixup_between/MNIST_cDCGAN_FID_log.txt','w+') as l_w:
    l_w.write(fid_log)
l_w.close()

min_fid = fid_hist['FID_score_mnist'][0]
min_epoch = 0
for epoch_fid in range(len(fid_hist['FID_score_mnist'])):
    fid = fid_hist['FID_score_mnist'][epoch_fid]
    if fid < min_fid:
        min_fid = fid
        min_epoch = epoch_fid
print("Lowest FID score to mnist: " + str(min_fid) + " @ epoch=" + str(fid_hist['epoch_num'][min_epoch]))