import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .utils import gabor_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    def forward(self, x):
        return x


class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)

        # Param instatiations
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase):
        random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        for i in range(self.out_channels):
            self.weight[i, random_channel[i]] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                             theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(self, sf, theta0, theta1, theta2, theta3, sigx, sigy, phase,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224):
        super().__init__()

        self.in_channels = 3

        self.simple_channels = 64
        self.complex_channels = 64
        self.out_channels = 128
        self.stride = stride
        self.input_size = input_size

        self.sf = sf
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta3
        self.theta3 = theta3
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level)
        self.fixed_noise = None

        self.simple_conv_q00 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q01 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q10 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q11 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q20 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q21 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q30 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q31 = GFB(self.in_channels, self.out_channels, ksize, stride)

        self.simple_conv_q00.initialize(sf=self.sf, theta=self.theta0, sigx=self.sigx, sigy=self.sigy,
                                        phase=self.phase)
        self.simple_conv_q01.initialize(sf=self.sf, theta=self.theta0, sigx=self.sigx, sigy=self.sigy,
                                        phase=self.phase + np.pi / 2)
        self.simple_conv_q10.initialize(sf=self.sf, theta=self.theta1, sigx=self.sigx, sigy=self.sigy,
                                        phase=self.phase)
        self.simple_conv_q11.initialize(sf=self.sf, theta=self.theta1, sigx=self.sigx, sigy=self.sigy,
                                        phase=self.phase + np.pi / 2)

        self.simple_conv_q20.initialize(sf=self.sf, theta=self.theta2, sigx=self.sigx, sigy=self.sigy,
                                        phase=self.phase)
        self.simple_conv_q21.initialize(sf=self.sf, theta=self.theta2, sigx=self.sigx, sigy=self.sigy,
                                        phase=self.phase + np.pi / 2)

        self.simple_conv_q30.initialize(sf=self.sf, theta=self.theta3, sigx=self.sigx, sigy=self.sigy,
                                        phase=self.phase)
        self.simple_conv_q31.initialize(sf=self.sf, theta=self.theta3, sigx=self.sigx, sigy=self.sigy,
                                        phase=self.phase + np.pi / 2)

        self.simple = nn.ReLU(inplace=True)
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = nn.ReLU(inplace=True)
        self.output = Identity()

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)

        # Noise [Batch, out_channels, H/stride, W/stride]
        x = self.noise_f(x)
        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.output(x)
        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q00(x)
        s_q1 = self.simple_conv_q01(x)
        s_q2 = self.simple_conv_q10(x)
        s_q3 = self.simple_conv_q11(x)
        s_q4 = self.simple_conv_q20(x)
        s_q5 = self.simple_conv_q21(x)
        s_q6 = self.simple_conv_q30(x)
        s_q7 = self.simple_conv_q31(x)
        c0 = self.complex(torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 +
                                     s_q1[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        s0 = self.simple(s_q0[:, 0:self.simple_channels, :, :])
        c1 = self.complex(torch.sqrt(s_q2[:, self.simple_channels:, :, :] ** 2 +
                                     s_q3[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        s1 = self.simple(s_q2[:, 0:self.simple_channels, :, :])
        c2 = self.complex(torch.sqrt(s_q4[:, self.simple_channels:, :, :] ** 2 +
                                     s_q5[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        s2 = self.simple(s_q4[:, 0:self.simple_channels, :, :])
        c3 = self.complex(torch.sqrt(s_q6[:, self.simple_channels:, :, :] ** 2 +
                                     s_q7[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))                                                                                                                                                                                  
        s3 = self.simple(s_q6[:, 0:self.simple_channels, :, :])
        layer1 = (self.k_exc * torch.cat((s0, c0),1))
        layer2 = (self.k_exc * torch.cat((s1, c1),1))
        layer3 = (self.k_exc * torch.cat((s2, c2),1))
        layer4 = (self.k_exc * torch.cat((s3, c3),1))
        layer_max =  torch.max(torch.tensor(layer3),torch.tensor(layer4))
        layer_max = torch.max(layer_max,torch.tensor(layer2))
        layer_max = torch.max(layer_max,torch.tensor(layer1))
        return self.gabors(layer_max)

    def noise_f(self, x):
        if self.noise_mode == 'neuronal':
            eps = 10e-5
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * torch.sqrt(F.relu(x.clone()) + eps)
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * \
                     torch.sqrt(F.relu(x.clone()) + eps)
            x -= self.noise_level
            x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
        noise_mean = torch.zeros(batch_size, self.out_channels, int(self.input_size / self.stride),
                                 int(self.input_size / self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(device)

    def unfix_noise(self):
        self.fixed_noise = None
