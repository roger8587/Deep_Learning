import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VAE(nn.Module):
    def __init__(self, in_channels = 3, hidden_dims=[16, 32, 64, 128, 256], z_dim=128):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(hidden_dims[-1], z_dim) # 均值 向量mu
        self.fc2 = nn.Linear(hidden_dims[-1], z_dim) # 向量var
        self.fc3 = nn.Linear(z_dim, hidden_dims[-1]) #decoder_input
        Encoder_modules = []
        for hdim in hidden_dims:
            Encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hdim,kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(hdim),
                    nn.LeakyReLU())
            )
            in_channels = hdim
        self.encoder = nn.Sequential(*Encoder_modules)
        hidden_dims.reverse()
        Decoder_modules = []
        for i in  range(len(hidden_dims) - 1):
            Decoder_modules.append(
                nn.Sequential(
                    #ConvTranspose2d 逆卷積
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size = 3,
                                       stride = 2,
                                       padding = 1,
                                       output_padding = 1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())                        
            )
        self.decoder = nn.Sequential(*Decoder_modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size = 3,
                                               stride = 2,
                                               padding = 1,
                                               output_padding = 1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels = 3,
                                      kernel_size = 3, padding = 1),
                            nn.Tanh())
    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        return self.fc1(result), self.fc2(result)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std) #normal分配
        return mu + eps * std

    def decode(self, z):
        result = self.fc3(z)
        result = result.view(-1, 256, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var