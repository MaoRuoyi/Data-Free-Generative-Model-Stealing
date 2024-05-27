'''

Step 1
cDCGAN AS Victim Model

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import torchvision
import os
from pytorch_fid import fid_score
import time
from shutil import copyfile
from pytorch_fid import fid_score
import matplotlib.pyplot as plt

batch_size = 128
train_epoch = 500
check_epoch = 10
picture_num_generate = 800
picture_num_calculate = 600
z_dimension = 100  # noise dimension

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ' + device)

#One-hot encode the target, use N 0s or 1s to binary encode N states
def onehot(x, class_count=10):
    return torch.eye(class_count)[x, :]

if not os.path.isdir('MNIST_cDCGAN_results'):
    os.mkdir('MNIST_cDCGAN_results')
if not os.path.isdir('MNIST_cDCGAN_results/Fixed_results'):
    os.mkdir('MNIST_cDCGAN_results/Fixed_results')
if not os.path.isdir('MNIST_cDCGAN_results/Backup'):
    os.mkdir('MNIST_cDCGAN_results/Backup')
if not os.path.isdir('g_img'):
    os.mkdir('g_img')

#Prepare training result storing dictionaries
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
fid_hist = {}
fid_hist['epoch_num'] = []
fid_hist['FID_score'] = []

#Prepare random noise input for epoch-based checking
fixed_z_ = torch.randn(batch_size, 100).to(device)
fixed_y_ = torch.randint(0, 10,size=(batch_size,))
fixed_y_label_ = onehot(fixed_y_).to(device)

#Data prepocessing
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

#Load built-in dataset MNIST 60000 training data
train_ds = torchvision.datasets.MNIST('../data',
                                     train=True,
                                     transform=img_transform)
                                     #download=True

#Divide in batches
dataloader = torch.utils.data.DataLoader(train_ds,
                                         batch_size=64,
                                         shuffle=True,
                                         drop_last=True)

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

#Plot loss-vs-epoch result figures
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

#Plot fid-vs-epoch result figures
def show_fid_hist(hist, show = False, save = False, path = 'FID_hist.png'):
    x = hist['epoch_num']
    y = hist['FID_score']

    plt.plot(x, y)

    plt.xlabel('Epoch')
    plt.ylabel('FID')

    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

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
        img = img.to(device)
        target = target.to(device)
        num_img = img.size(0)
        real_img = img
        target = F.one_hot(target, num_classes = 10)
        target = target.to(torch.float32) #To fix RuntimeError: mat1 and mat2 must have the same dtype
 
        #Train discriminator
        real_out = D(real_img,target)
        real_label = torch.ones_like(real_out)
        
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out
 
        z = torch.randn(num_img, z_dimension, device=device)
        fake_img = G(z,target)
        fake_out = D(fake_img.detach(),target.detach())
        fake_label = torch.zeros_like(fake_out)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out
 
        d_loss = (d_loss_real + d_loss_fake)/2
        d_optimizer.zero_grad()
        d_loss.backward()
        if d_loss.item() > 0.1:
            d_optimizer.step()
        D_losses.append(d_loss.item())
 
        #Train generator
        output = D(fake_img,target)
        g_loss = criterion(output, real_label)     
 
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        G_losses.append(g_loss.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    
    G.eval()
    with torch.no_grad():
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
        
        fixed_p = 'MNIST_cDCGAN_results/Fixed_results/MNIST_cDCGAN_' + str(epoch + 1) + '.png'
        test_images = G(fixed_z_, fixed_y_label_)
        test_images = (test_images+1)/2
        save_image(test_images.clamp(0,1), fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        #Save output images every few epochs
        if (epoch+1)%check_epoch == 0:
            torch.save(G, 'MNIST_cDCGAN_results/Backup/generator_epoch_{}.pth'.format(epoch+1))
            torch.save(D, 'MNIST_cDCGAN_results/Backup/discriminator_epoch_{}.pth'.format(epoch+1))
            if not os.path.exists('./g_img/epoch_'+str(epoch+1)):
                os.mkdir('./g_img/epoch_'+str(epoch+1))
            for label in range(10):
                label_list = []
                label_list.append(label)
                label_input = torch.tensor(label_list)
                label_input_onehot=torch.eye(10)[label_input, :]

                for g in range(picture_num_generate):
                    if not os.path.exists('./g_img/epoch_'+str(epoch+1)+'/'+str(label)):
                        os.mkdir('./g_img/epoch_'+str(epoch+1)+'/'+str(label))
                    if not os.path.exists('./g_img/epoch_'+str(epoch+1)+'/g_all'):
                        os.mkdir('./g_img/epoch_'+str(epoch+1)+'/g_all')
                    test_input = torch.randn(1, 100, device = device)
                    prediction = G(test_input, label_input_onehot.to(device))
                    prediction_img = (prediction+1)/2
                    save_image(prediction_img.clamp(0,1), './g_img/epoch_{}/{}/fake_images_{}_{}.png'.format(epoch+1,label,label,g+1), nrow=1, format='png')
                    if g < picture_num_calculate:
                        copyfile('./g_img/epoch_{}/{}/fake_images_{}_{}.png'.format(epoch+1,label,label,g+1), './g_img/epoch_{}/g_all/fake_images_{}_{}.png'.format(epoch+1,label,g+1))
            print("Image of epoch "+str(epoch+1)+" saved.")

            #Calculate fid scores to MNIST and to victim output and save in dictionary
            fid_value = fid_score.calculate_fid_given_paths(['./g_img/epoch_{}/g_all'.format(epoch+1),
                                                            '../o_img_6000'],
                                                            batch_size=1,
                                                            dims = 2048,
                                                            device = device,
                                                            num_workers=1
                                                            )
            print('FID: '+str(fid_value))
            fid_hist['epoch_num'].append(epoch+1)
            fid_hist['FID_score'].append(fid_value)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G, "MNIST_cDCGAN_results/generator_param.pth")#torch.save()函数用于保存模型的状态字典，将模型的每一层映射到其对应的张量参数（权重和偏差）作为键值对，D.state_dict()返回每个模型的当前状态字典。状态字典包含模型的所有学习参数，可用于恢复训练或稍后加载模型以进行推理
torch.save(D, "MNIST_cDCGAN_results/discriminator_param.pth")

#Plot result figures and find lowest FID
show_train_hist(train_hist, save=True, path='MNIST_cDCGAN_results/MNIST_cDCGAN_train_hist.png')
show_fid_hist(fid_hist, save=True, path='MNIST_cDCGAN_results/MNIST_cDCGAN_FID_hist.png')
min_fid = fid_hist['FID_score'][0]
min_epoch = 0
for epoch_fid in range(len(fid_hist['FID_score'])):
    fid = fid_hist['FID_score'][epoch_fid]
    if fid < min_fid:
        min_fid = fid
        min_epoch = epoch_fid
print("Final FID score: " + str(fid_value))
print("Lowest FID score: " + str(min_fid) + " @ epoch=" + str(fid_hist['epoch_num'][min_epoch]))