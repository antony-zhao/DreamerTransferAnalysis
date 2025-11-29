from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

dataset = load_dataset('Maysee/tiny-imagenet', split='train')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x - 0.5),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), # some images are grayscale for some reason, idk  
])

class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

dataloader = DataLoader(HuggingFaceDataset(dataset, transform), batch_size=512, shuffle=True)
"""
encoder.cnn_encoder.model.0._model.0.weight torch.Size([96, 3, 4, 4])
encoder.cnn_encoder.model.0._model.1.weight torch.Size([96])
encoder.cnn_encoder.model.0._model.1.bias torch.Size([96])
encoder.cnn_encoder.model.0._model.3.weight torch.Size([192, 96, 4, 4])
encoder.cnn_encoder.model.0._model.4.weight torch.Size([192])
encoder.cnn_encoder.model.0._model.4.bias torch.Size([192])
encoder.cnn_encoder.model.0._model.6.weight torch.Size([384, 192, 4, 4])
encoder.cnn_encoder.model.0._model.7.weight torch.Size([384])
encoder.cnn_encoder.model.0._model.7.bias torch.Size([384])
encoder.cnn_encoder.model.0._model.9.weight torch.Size([768, 384, 4, 4])
encoder.cnn_encoder.model.0._model.10.weight torch.Size([768])
encoder.cnn_encoder.model.0._model.10.bias torch.Size([768])
observation_model.cnn_decoder.model.0.weight torch.Size([12288, 5120])
observation_model.cnn_decoder.model.0.bias torch.Size([12288])
observation_model.cnn_decoder.model.2._model.0.weight torch.Size([768, 384, 4, 4])
observation_model.cnn_decoder.model.2._model.1.weight torch.Size([384])
observation_model.cnn_decoder.model.2._model.1.bias torch.Size([384])
observation_model.cnn_decoder.model.2._model.3.weight torch.Size([384, 192, 4, 4])
observation_model.cnn_decoder.model.2._model.4.weight torch.Size([192])
observation_model.cnn_decoder.model.2._model.4.bias torch.Size([192])
observation_model.cnn_decoder.model.2._model.6.weight torch.Size([192, 96, 4, 4])
observation_model.cnn_decoder.model.2._model.7.weight torch.Size([96])
observation_model.cnn_decoder.model.2._model.7.bias torch.Size([96])
observation_model.cnn_decoder.model.2._model.9.weight torch.Size([96, 3, 4, 4])
observation_model.cnn_decoder.model.2._model.9.bias torch.Size([3])
"""

def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

class ChannelNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Encoder(nn.Module):
    def __init__(self, base=96):
        super().__init__()
        layers = []
        for i in range(4):
            layers.append(nn.Conv2d(3 if i == 0 else base * 2 ** (i - 1), base * 2 ** i, 4, 2, 1, bias=False))
            layers.append(ChannelNorm(base * 2 ** i))
            layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)
        self.linear = nn.Linear(12288, 1024)
    
    def forward(self, x):
        x = self.layers(x)
        return self.linear(x.flatten(-3))

class Decoder(nn.Module):
    def __init__(self, base=96):
        super().__init__()
        self.linear = nn.Linear(1024, 12288)
        layers = []
        for i in reversed(range(4)):
            layers.append(nn.ConvTranspose2d(base * 2 ** i, 3 if i == 0 else base * 2 ** (i - 1), 4, 2, 1, bias=i == 0))
            if i > 0:
                layers.append(ChannelNorm(base * 2 ** (i - 1)))
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.unflatten(x, 1, (-1, 4, 4))
        return self.layers(x)

encoder = Encoder().to(device)
decoder = Decoder().to(device)
opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 1e-4)
losses = []
for _ in range(20000):
    images, _ = next(iter(dataloader))
    images = images.to(device)
    latent = encoder(images)
    reconstruction = decoder(latent)
    loss = nn.functional.mse_loss(images, reconstruction)
    losses.append(loss.detach().cpu().numpy())
    loss.backward()
    opt.step()
    opt.zero_grad()
plt.plot(losses)
plt.savefig("pretraining.png")
torch.save(encoder.state_dict(), "pretrained_encoder.ckpt")
torch.save(decoder.state_dict(), "pretrained_decoder.ckpt")