import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class AE(nn.Module):
    def __init__(self, in_channels = 1, hidden_dims=[128, 256, 512], z_dim=128):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(12800, z_dim) # 均值 向量mu
        self.fc2 = nn.Linear(z_dim, 12800) #decoder_input
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hidden_dims[0],kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size = 2, return_indices=True))
        self.conv2 = nn.Sequential(
                    nn.Conv2d(hidden_dims[0], out_channels=hidden_dims[1],kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(hidden_dims[1]),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size = 2, return_indices=True))
        self.conv3 = nn.Sequential(
                    nn.Conv2d(hidden_dims[1], out_channels=hidden_dims[2],kernel_size= 3, padding  = 1),
                    nn.BatchNorm2d(hidden_dims[2]),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size = 2, return_indices=True))
        self.upsample = nn.MaxUnpool2d(2)
        #ConvTranspose2d 逆卷積
        
        self.trans1 = nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[2],
                                       hidden_dims[1],
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1,
                                       output_padding = 0),
                    nn.BatchNorm2d(hidden_dims[1]),
                    nn.LeakyReLU())
        self.trans2 = nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[1],
                                       hidden_dims[0],
                                       kernel_size = 3,
                                       stride = 2,
                                       padding = 1,
                                       output_padding = 1),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.LeakyReLU())
        self.trans3 = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[0],
                                               hidden_dims[0],
                                               kernel_size = 3,
                                               stride = 2,
                                               padding = 1,
                                               output_padding = 1),
                            nn.BatchNorm2d(hidden_dims[0]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[0], out_channels = 1,
                                      kernel_size = 3, padding = 1),
                            nn.Tanh())
    
    def latent_code(self,x):
        result = torch.flatten(x, start_dim=1)
        return self.fc1(result)
    
    def forward(self, x):
        #print(x.size())
        x, indice1 = self.conv1(x)
        #print(x.size())
        x, indice2 = self.conv2(x)
        #print(x.size())
        x, indice3 = self.conv3(x)
        #print(x.size())
        x = self.upsample(x,indice3)
        #print(x.size())
        x = self.trans1(x)
        #print(x.size())
        x = self.upsample(x,indice2)
        x = self.trans2(x)
        x = self.upsample(x,indice1)
        x = self.trans3(x)
        return x

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.data import DataLoader
from IPython.display import Image
from IPython.core.display import Image, display
from dae import AE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-3
batch_size = 32
num_epoch = 150
transform =  transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5])
        ])
data = datasets.ImageFolder('./normal',transform = transform)
dataloader = DataLoader(data, batch_size = batch_size, shuffle = True)
len(data.imgs), len(dataloader)
model = AE().to(device)
model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0001)
criterion = nn.MSELoss(reduction='sum')
Loss = []
for epoch in range(num_epoch):
    num_data = 0
    total_loss = 0
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        #print(images.size())
        batch_num = len(images)
        num_data += batch_num
        recon_images = model(images)
        loss = criterion(recon_images, images)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    Loss.append(total_loss)
    if (epoch+1) % 10 == 0:
        print('Train epoch: {}/{} Training Loss: {}'.format(epoch+1, num_epoch, total_loss/num_data))
invTrans = transforms.Compose([ 
    transforms.Normalize(mean = [0.],std = [ 1/0.5]),
    transforms.Normalize(mean = [ -0.5],std = [ 1.]),
    ])
torch.save(model.state_dict(), 'params130.pkl')
pretrained_dict = torch.load('params130.pkl')
model_dict = model.state_dict()

#%%
from PIL import Image as pic
test_image = pic.open('./normal/normal/Defect_P0000_P0135_000_000.jpg')
test_image
fixed_x = transform(test_image)
model.load_state_dict(torch.load('params130.pkl', map_location=torch.device('cpu')))
model.eval()
save_image(invTrans(fixed_x).data, 'sample_image.png')
display(Image('sample_image.png', width=160, unconfined=True))
fixed_x = fixed_x.to(device)
recon_x= model(fixed_x.view(-1,1,160,160))
recon_x = invTrans(recon_x.view(1,160,160))
save_image(recon_x.data, 'recon_image.png')
display(Image('recon_image.png', width=160, unconfined=True))

#https://arxiv.org/pdf/2008.12589.pdf
