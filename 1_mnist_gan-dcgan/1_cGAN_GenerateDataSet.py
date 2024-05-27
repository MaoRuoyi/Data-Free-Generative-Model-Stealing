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

batch_size = 128
train_epoch = 1000
check_epoch = 10
picture_num_generate = 1000
picture_num_calculate = 600

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ' + device)

if not os.path.isdir('MNIST_cGAN_results'):
    os.mkdir('MNIST_cGAN_results')
if not os.path.isdir('MNIST_cGAN_results/Fixed_results'):
    os.mkdir('MNIST_cGAN_results/Fixed_results')
if not os.path.isdir('MNIST_cGAN_results/Backup'):
    os.mkdir('MNIST_cGAN_results/Backup')
if not os.path.isdir('g_img'):
    os.mkdir('g_img')

#One-hot encode the target, use N 0s or 1s to binary encode N states
def onehot(x, class_count=10):
    return torch.eye(class_count)[x, :]

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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
    ])

#Load built-in dataset MNIST 60000 training data
train_ds = torchvision.datasets.MNIST('../data',
                                     train=True,
                                     transform=transform,
                                     target_transform=onehot)
                                     #download=True

#Divide in batches
dataloader = torch.utils.data.DataLoader(train_ds,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True) 
 
#Define generator with 100-length noise and 10-length target as inputs, [batchsize, 1, 28, 28] image tensor as output
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(110, 128),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(128, 256),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(256, 512),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(512, 768),
                                 nn.BatchNorm1d(768),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(768, 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(1024, 28 * 28), 
                                 nn.Tanh()
                                 )
    def forward(self, x, tar):
        input = torch.cat([x, tar], 1)
        input = self.gen(input)
        input = input.view(-1, 1, 28, 28)
        return input
 
#Define discriminator with the image tensor and 10-length target as input and a binary classification as output
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.linear = nn.Sequential(nn.Linear(10, 28*28),nn.Sigmoid())
        self.disc = nn.Sequential(nn.Linear(28*28, 512),
                                 nn.LeakyReLU(), 
                                 nn.Linear(512, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, 1),
                                 nn.Sigmoid() 
        )
        self.cat = nn.Linear(2*28*28,1*28*28)
    def forward(self, x, tar):
        tar = self.linear(tar)
        tar = tar.view(-1,1,28,28)
        input = torch.cat([x, tar], 1)
        input = input.view(-1, 2*28*28)
        input = self.cat(input)
        input = self.disc(input)
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
print('Number of learnable params of Generator: {}.'.format(num_total_learnable_params_G))      #G: 2169232
num_total_learnable_params_D = sum(i.numel() for i in D.parameters() if i.requires_grad)
print('Number of learnable params of Discriminator: {}.'.format(num_total_learnable_params_D))  #D: 1772225
 
d_optim = torch.optim.Adam(D.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(G.parameters(), lr=0.0001)
 
loss_function = torch.nn.BCELoss()

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G.train()#Enter training mode
    D_losses = []
    G_losses = []

    epoch_start_time = time.time()
    for step, (img, target) in enumerate(dataloader):
        img = img.to(device)
        target = target.to(device)
        size = img.size(0)
 
        random_noise = torch.randn(size, 100, device=device)
 
        #Train the discriminator
        d_optim.zero_grad()
        real_output = D(img,target)
        d_real_loss = loss_function(real_output,
                                    torch.ones_like(real_output))
        d_real_loss.backward()
 
        gen_img = G(random_noise,target)
        fake_output = D(gen_img.detach(),target.detach())
        d_fake_loss = loss_function(fake_output,
                                    torch.zeros_like(fake_output))
        d_loss = (d_real_loss + d_fake_loss)/2
        d_fake_loss.backward()
        d_optim.step()
        D_losses.append(d_loss.item())

        #Train the generator
        g_optim.zero_grad()
        fake_output = D(gen_img,target)
        g_loss = loss_function(fake_output,
                               torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()
        G_losses.append(g_loss.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    
    G.eval()#Enter evaluation mode
    with torch.no_grad():
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
        
        fixed_p = 'MNIST_cGAN_results/Fixed_results/MNIST_cGAN_' + str(epoch + 1) + '.png'
        test_images = G(fixed_z_, fixed_y_label_)
        test_images = (test_images+1)/2
        save_image(test_images.clamp(0,1), fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        #Save output images every few epochs
        if (epoch+1)%check_epoch == 0:
            torch.save(G, 'MNIST_cGAN_results/Backup/generator_epoch_{}.pth'.format(epoch+1))
            torch.save(D, 'MNIST_cGAN_results/Backup/discriminator_epoch_{}.pth'.format(epoch+1))
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

            #Calculate fid score and save in dictionary
            if epoch > 20: #Result too big at first, out of range
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
torch.save(G, "MNIST_cGAN_results/generator_param.pth")
torch.save(D, "MNIST_cGAN_results/discriminator_param.pth")

#Plot result figures and find lowest FID
show_train_hist(train_hist, save=True, path='MNIST_cGAN_results/MNIST_cGAN_train_hist.png')
show_fid_hist(fid_hist, save=True, path='MNIST_cGAN_results/MNIST_cGAN_FID_hist.png')
min_fid = fid_hist['FID_score'][0]
min_epoch = 0
for epoch_fid in range(len(fid_hist['FID_score'])):
    fid = fid_hist['FID_score'][epoch_fid]
    if fid < min_fid:
        min_fid = fid
        min_epoch = epoch_fid
print("Final FID score: " + str(fid_value))
print("Lowest FID score: " + str(min_fid) + " @ epoch=" + str(fid_hist['epoch_num'][min_epoch]))
