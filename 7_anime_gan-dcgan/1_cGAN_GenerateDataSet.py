'''

Step 1
cGAN AS Victim Model

'''

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import os
import time
from shutil import copyfile
from pytorch_fid import fid_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

batch_size = 128
train_epoch = 1000
check_epoch = 10
picture_num_generate = 6000
picture_num_calculate = 6000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ' + device)

if not os.path.isdir('Anime_cGAN_results'):
    os.mkdir('Anime_cGAN_results')
if not os.path.isdir('Anime_cGAN_results/Fixed_results'):
    os.mkdir('Anime_cGAN_results/Fixed_results')
if not os.path.isdir('Anime_cGAN_results/Backup'):
    os.mkdir('Anime_cGAN_results/Backup')
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
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(3*100, 3*400),
                                 nn.BatchNorm1d(3*400),
                                 nn.ReLU(),
                                 nn.Linear(3*400, 3*800),
                                 nn.BatchNorm1d(3*800),
                                 nn.ReLU(),
                                 nn.Linear(3*800, 3*28 * 28),
                                 nn.Tanh()
                                 )
 
    def forward(self, x): 
        input = self.gen(x)
        input = input.view(-1, 3, 28, 28)
        return input
 
#Define discriminator with the image tensor as input and a binary classification as output
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.disc1 = nn.Sequential(nn.Linear(3*28*28, 3*256),
                                 nn.LeakyReLU(0.2, inplace=True)
        )
        self.disc2 = nn.Sequential(nn.Linear(3*256, 3*64),
                                 nn.LeakyReLU(0.2, inplace=True)
        )
        self.disc3 = nn.Sequential(nn.Linear(3*64, 1),
                                 nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        input = x.view(-1, 3*28*28)
        input = self.disc1(input)
        input = self.disc2(input)
        input = self.dropout(input)
        input = self.disc3(input)
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

G = Generator().to(device)
D = Discriminator().to(device)

num_total_learnable_params_G = sum(i.numel() for i in G.parameters() if i.requires_grad)
print('Number of learnable params of Generator: {}.'.format(num_total_learnable_params_G))      #G: 5031984
num_total_learnable_params_D = sum(i.numel() for i in D.parameters() if i.requires_grad)
print('Number of learnable params of Discriminator: {}.'.format(num_total_learnable_params_D))  #D: 3185041

d_optim = torch.optim.Adam(D.parameters(), lr=0.00008, betas=(0.5, 0.999))
g_optim = torch.optim.Adam(G.parameters(), lr=0.0008, betas=(0.5, 0.999))
 
loss_function = torch.nn.BCELoss()

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G.train()#Enter training mode
    D.train()
    D_losses = []
    G_losses = []

    epoch_start_time = time.time()
    for step, (img) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
 
        random_noise = torch.randn(size, 300, device=device)

        #Train the discriminator
        d_optim.zero_grad()
        real_output = D(img)
        d_real_loss = loss_function(real_output,
                                    torch.ones_like(real_output))
 
        gen_img = G(random_noise)
        fake_output = D(gen_img.detach())
        d_fake_loss = loss_function(fake_output,
                                    torch.zeros_like(fake_output)) 
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        if d_loss.item()>0.1:
            d_optim.step()
        D_losses.append(d_loss.item())

        #Train the generator
        g_optim.zero_grad()
        fake_output = D(gen_img)
        g_loss = loss_function(fake_output,
                               torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()
        G_losses.append(g_loss.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    
    G.eval()
    with torch.no_grad():
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
        
        fixed_p = 'Anime_cGAN_results/Fixed_results/Anime_cGAN_' + str(epoch + 1) + '.png'
        test_images = G(fixed_z_)
        test_images = (test_images+1)/2
        save_image(test_images.clamp(0,1), fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        #Save output images every few epochs
        if (epoch+1)%check_epoch == 0:
            torch.save(G, 'Anime_cGAN_results/Backup/generator_epoch_{}.pth'.format(epoch+1))
            torch.save(D, 'Anime_cGAN_results/Backup/discriminator_epoch_{}.pth'.format(epoch+1))
            if not os.path.exists('./g_img/epoch_'+str(epoch+1)):
                os.mkdir('./g_img/epoch_'+str(epoch+1))

            for g in range(picture_num_generate):
                if not os.path.exists('./g_img/epoch_'+str(epoch+1)+'/g_all'):
                    os.mkdir('./g_img/epoch_'+str(epoch+1)+'/g_all')
                test_input = torch.randn(1, 300, device = device)
                prediction = G(test_input)
                prediction_img = (prediction+1)/2
                save_image(prediction_img.clamp(0,1), './g_img/epoch_{}/fake_images_{}.png'.format(epoch+1,g+1), nrow=1, format='png')
                if g < picture_num_calculate:
                    copyfile('./g_img/epoch_{}/fake_images_{}.png'.format(epoch+1,g+1), './g_img/epoch_{}/g_all/fake_images_{}.png'.format(epoch+1,g+1))
            print("Image of epoch "+str(epoch+1)+" saved.")
                
            #Calculate fid score and save in dictionary
            fid_value = fid_score.calculate_fid_given_paths(['./g_img/epoch_{}/g_all'.format(epoch+1),
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
torch.save(G, "Anime_cGAN_results/generator_param.pth")
torch.save(D, "Anime_cGAN_results/discriminator_param.pth")

#Plot result figures
show_train_hist(train_hist, save=True, path='Anime_cGAN_results/Anime_cGAN_train_hist.png')
show_fid_hist(fid_hist, save=True, path='Anime_cGAN_results/Anime_cGAN_FID_hist.png')

#Save results in log for backup
fid_log = "Epoch = " + str(fid_hist['epoch_num']) + "\n\nAnime = " + str(fid_hist['FID_score'])
with open('Anime_cGAN_results/Anime_cGAN_FID_log.txt','w+') as l_w:
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
