'''

Step 1
cDCGAN AS Victim Model

'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import torchvision
import os
from PIL import Image
from pytorch_fid import fid_score
import time
from shutil import copyfile
from pytorch_fid import fid_score
import matplotlib.pyplot as plt
from PIL import Image

batch_size = 128
train_epoch = 1000
check_epoch = 10
picture_num_generate = 6000
picture_num_calculate = 6000
z_dimension = 300  # noise dimension

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ' + device)

if not os.path.isdir('Anime_cDCGAN_results'):
    os.mkdir('Anime_cDCGAN_results')
if not os.path.isdir('Anime_cDCGAN_results/Fixed_results'):
    os.mkdir('Anime_cDCGAN_results/Fixed_results')
if not os.path.isdir('Anime_cDCGAN_results/Backup'):
    os.mkdir('Anime_cDCGAN_results/Backup')
if not os.path.isdir('dc_img'):
    os.mkdir('dc_img')

#Prepare training result storing dictionaries
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
fid_hist = {}
fid_hist['epoch_num'] = []
fid_hist['FID_score'] = []

#Prepare random noise input for epoch-based checking, no target needed
fixed_z_ = torch.randn(batch_size, 300).to(device)

#Data prepocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #Three channels colored images
    ])

#Define loading of self-built dataset from a folder storing randomly selected 60000 images
class MyDataset(Dataset):
    def __init__(self, path, transform = None):
        self.path = path
        self.imgs = os.listdir(self.path)
        self.transform = transform
        self.len = len(self.imgs)
    def __getitem__(self, index):
        image_index = self.imgs[index]
        img_path = os.path.join(self.path, image_index)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img
    def __len__(self):
        return self.len

dataset=MyDataset(path = "../o_img_60000_anime", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
#Define generator with 300-length noise as inputs, [batchsize, 3, 28, 28] image tensor as output
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc = nn.Linear(300, 3*784)
        self.br = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.ConvTranspose2d(3, 50, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.ConvTranspose2d(50, 25, 3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.ConvTranspose2d(25, 3, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
 
    def forward(self, x):
        input = self.fc(x)
        input = input.view(input.size(0), 3, 28, 28)
        input = self.downsample1(input)
        input = self.downsample2(input)
        input = self.downsample3(input)
        return input

#Define discriminator with the image tensor as input and a binary classification as output
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=True),
            nn.Sigmoid(),
        ) 
    def forward(self, x):
        input = torch.reshape(x,(-1, 3, 28, 28))
        input = self.model(input)
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
print('Number of learnable params of Generator: {}.'.format(num_total_learnable_params_G))      #G: 721389
num_total_learnable_params_D = sum(i.numel() for i in D.parameters() if i.requires_grad)
print('Number of learnable params of Discriminator: {}.'.format(num_total_learnable_params_D))  #D: 52729026
 
criterion = nn.BCELoss()
 
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0006, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.006, betas=(0.5, 0.999))
 
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G.train()#Enter training mode
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for i, (img) in enumerate(dataloader):
        img = img.to(device)
        num_img = img.size(0)
        real_img = img

        #Train the discriminator
        real_out = D(real_img)
        real_label = torch.ones_like(real_out)
        
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out

        z = torch.randn(num_img, z_dimension, device=device)
        fake_img = G(z)
        fake_out = D(fake_img.detach())
        fake_label = torch.zeros_like(fake_out)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out
 
        d_loss = (d_loss_real + d_loss_fake)/2
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        D_losses.append(d_loss.item())
 
        #Train generator
        output = D(fake_img)
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
        
        fixed_p = 'Anime_cDCGAN_results/Fixed_results/Anime_cDCGAN_' + str(epoch + 1) + '.png'
        test_images = G(fixed_z_)
        test_images = (test_images+1)/2
        save_image(test_images.clamp(0,1), fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        #Save output images every few epochs
        if (epoch+1)%check_epoch == 0:
            torch.save(G, 'Anime_cDCGAN_results/Backup/generator_epoch_{}.pth'.format(epoch+1))
            torch.save(D, 'Anime_cDCGAN_results/Backup/discriminator_epoch_{}.pth'.format(epoch+1))
            if not os.path.exists('./dc_img/epoch_'+str(epoch+1)):
                os.mkdir('./dc_img/epoch_'+str(epoch+1))
            for g in range(picture_num_generate):
                if not os.path.exists('./dc_img/epoch_'+str(epoch+1)+'/g_all'):
                    os.mkdir('./dc_img/epoch_'+str(epoch+1)+'/g_all')
                test_input = torch.randn(1, 300, device = device)
                prediction = G(test_input)
                prediction_img = (prediction+1)/2
                save_image(prediction_img.clamp(0,1), './dc_img/epoch_{}/fake_images_{}.png'.format(epoch+1,g+1), nrow=1, format='png')
                if g < picture_num_calculate:
                    copyfile('./dc_img/epoch_{}/fake_images_{}.png'.format(epoch+1,g+1), './dc_img/epoch_{}/g_all/fake_images_{}.png'.format(epoch+1,g+1))
            print("Image of epoch "+str(epoch+1)+" saved.")

            #Calculate fid score and save in dictionary
            fid_value = fid_score.calculate_fid_given_paths(['./dc_img/epoch_{}/g_all'.format(epoch+1),
                                                            '../o_img_28_6000_anime'],
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
torch.save(G, "Anime_cDCGAN_results/generator_param.pth")
torch.save(D, "Anime_cDCGAN_results/discriminator_param.pth")

#Plot result figures
show_train_hist(train_hist, save=True, path='Anime_cDCGAN_results/Anime_cDCGAN_train_hist.png')
show_fid_hist(fid_hist, save=True, path='Anime_cDCGAN_results/Anime_cDCGAN_FID_hist.png')

#Save results in log for backup
fid_log = "Epoch = " + str(fid_hist['epoch_num']) + "\n\nAnime = " + str(fid_hist['FID_score'])
with open('Anime_cDCGAN_results/Anime_cDCGAN_FID_log.txt','w+') as l_w:
    l_w.write(fid_log)
l_w.close()

#Find lowest FID
min_fid = fid_hist['FID_score'][0]
min_epoch = 0
for epoch_fid in range(len(fid_hist['FID_score'])):
    fid = fid_hist['FID_score'][epoch_fid]
    if fid < min_fid:
        min_fid = fid
        min_epoch = epoch_fid
print("Final FID score: " + str(fid_value))
print("Lowest FID score: " + str(min_fid) + " @ epoch=" + str(fid_hist['epoch_num'][min_epoch]))
