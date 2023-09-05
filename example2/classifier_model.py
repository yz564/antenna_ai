import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    """A very simple simulation model."""

    def __init__(self, input_space: int, pred_space: int, hidden_size: int):
        super().__init__()
        self.affine1 = nn.Linear(input_space, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.affine3 = nn.Linear(hidden_size, hidden_size)
        self.affine4 = nn.Linear(hidden_size, hidden_size)

        self.pred_head = nn.Linear(hidden_size, pred_space)

    def forward(self, x):
        x = torch.relu(self.affine1(x))
        x = torch.relu(self.affine2(x))
        x = torch.relu(self.affine3(x))
        x = torch.relu(self.affine4(x))

        # x = torch.sin(self.affine1(x))
        # x = torch.sin(self.affine2(x))
        # x = torch.sin(self.affine3(x))
        # x = torch.sin(self.affine4(x))

        return self.pred_head(x)

class CoordConvLayer(nn.Module):
    """A CNN"""
    def __init__(self, input_channels, output_channels, kernel_size, stride, height, width, padding=0, fourier_features=None):

        super().__init__()
        x = (torch.arange(0, width) / width).expand(height, -1)
        y = (torch.arange(0, height).unsqueeze(1) / height).expand(-1, width)
        coords = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).to(dtype=torch.float32)
        self.coords = nn.Parameter(coords, requires_grad=False)

        self.fourier_features = fourier_features
        if self.fourier_features:
            self.layer = nn.Conv2d(self.fourier_features.B_height * 2, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, dtype=torch.float32)
        else:
            self.layer = nn.Conv2d(input_channels + 2, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, dtype=torch.float32)

    def forward(self, inp):
        expanded_coords = self.coords.expand(inp.shape[0], -1, -1, -1)
        coord_inp = torch.cat([inp, expanded_coords], dim=1)
        if self.fourier_features:
            coord_inp = self.fourier_features(coord_inp)
        return self.layer(coord_inp)

class ConvModel(nn.Module):
    """A CNN"""
    def __init__(self, pred_space: int, hidden_size: int, backbone_file: str=None):
        super().__init__()
        if backbone_file:
            self.freeze = True
            self.encoder = torch.load(backbone_file)
            print(f"Loaded backbone from {backbone_file}")
        else:
            self.freeze = False
            encoder = []
            encoder.append(CoordConvLayer(1, 64, (2,4), 1, 50, 220, padding=1))
            encoder.append(nn.GELU())
            encoder.append(nn.MaxPool2d(kernel_size=(2,4), stride=2))
            
            encoder.append(nn.Conv2d(64, 64, kernel_size=(2,4), stride=1, padding=1))
            encoder.append(nn.GELU())
            encoder.append(nn.MaxPool2d(kernel_size=(2,4), stride=2))

            encoder.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            encoder.append(nn.GELU())
            encoder.append(nn.MaxPool2d(kernel_size=4, stride=2))

            encoder.append(nn.Flatten())
            '''
            encoder.append(nn.Linear(8000, hidden_size))
            encoder.append(nn.SiLU())
            encoder.append(nn.Dropout(p=.5))
            '''
            encoder.append(nn.Linear(8000, hidden_size))
            encoder.append(nn.SiLU())
            encoder.append(nn.Dropout(p=.5))
            # add more hidden layers
            encoder.append(nn.Linear(hidden_size, hidden_size//2))
            encoder.append(nn.SiLU())
            encoder.append(nn.Dropout(p=.5))
            hidden_size=hidden_size//2
            encoder.append(nn.Linear(hidden_size, hidden_size//2))
            encoder.append(nn.SiLU())
            encoder.append(nn.Dropout(p=.5))
            hidden_size=hidden_size//2
            
            self.encoder = nn.Sequential(*encoder)
        self.freq_resp = nn.Sequential(nn.Linear(hidden_size, pred_space),nn.Sigmoid())
        #self.freq_resp = nn.Linear(hidden_size, pred_space)
        #self.viability = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid(),nn.Linear(1,1)) #sigmoid output range is (0,1)
        self.viability = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                enc = self.encoder(x)
        else:
            enc = self.encoder(x)
            freq_response = self.freq_resp(enc)
            score = self.viability(enc)
        return freq_response, score
