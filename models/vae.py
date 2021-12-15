import torch
import torch.nn as nn
import torch.nn.functional as F

from env import n_fft, n_mfcc_model, frame_size


class ToyAE(nn.Module):
    def __init__(self, frame_size=n_fft // 2 + 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(frame_size, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, frame_size),
        )

    def forward(self, x):
        return self.model(x)


class MultiFeaturePerceptron(nn.Module):
    def __init__(self, frame_size=n_fft // 2 + 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(frame_size + n_mfcc_model - 1 + frame_size // 4, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, frame_size),
        )

    def forward(self, x):
        return self.model(x)


class OldCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input
        self.frame_size = frame_size
        self.n_mfcc = n_mfcc_model - 1
        # self.n_mod_ceps = frame_size // 2 + 1
        self.n_mod_ceps = 85

        # model

        self.formant_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1,
                      kernel_size=(16,), stride=(2,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=1, out_channels=1,
                      kernel_size=(8,), stride=(4,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(125, 123),
            nn.ConvTranspose1d(in_channels=1, out_channels=1,
                               output_padding=(1,),
                               kernel_size=(8,), stride=(2,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose1d(in_channels=1, out_channels=1,
                               output_padding=(1,),
                               kernel_size=(16,), stride=(4,))
        )

        self.fundamental_freq_model = nn.Sequential(
            nn.Linear(self.n_mod_ceps + self.n_mfcc, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, frame_size),
        )

    def forward(self, x):
        # split
        log_stft = x[:, :frame_size]
        mfcc = x[:, frame_size:frame_size + self.n_mfcc]
        fundamental_freq_dat = x[:, frame_size + self.n_mfcc
                                    :frame_size + self.n_mfcc + self.n_mod_ceps]
        filtered_formant = x[:, frame_size + self.n_mfcc + self.n_mod_ceps:]
        return torch.cat(
            (
                self.formant_model(filtered_formant.view(
                    -1, 1, frame_size)).view(-1, frame_size),
                self.fundamental_freq_model(
                    torch.cat((fundamental_freq_dat, mfcc), 1))
            ), 1)


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input
        self.frame_size = frame_size
        self.n_mfcc = n_mfcc_model - 1
        # self.n_mod_ceps = frame_size // 2 + 1
        self.n_mod_ceps = 85

        # model

        self.formant_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3,
                      kernel_size=(16,), stride=(2,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=5,
                      kernel_size=(8,), stride=(4,)),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.formant_transform = nn.Sequential(
             nn.Linear(125*5 + self.n_mfcc, 123*5),
        )

        self.formant_decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=5, out_channels=3,
                               output_padding=(1,),
                               kernel_size=(8,), stride=(2,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose1d(in_channels=3, out_channels=1,
                               output_padding=(1,),
                               kernel_size=(16,), stride=(4,))
        )

        self.fundamental_freq_model = nn.Sequential(
            nn.Linear(self.n_mod_ceps + self.n_mfcc, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, frame_size),
        )

    def forward(self, x):
        n_samples = x.shape[0]
        # split
        log_stft = x[:, :frame_size]
        mfcc = x[:, frame_size:frame_size + self.n_mfcc]
        fundamental_freq_dat = x[:, frame_size + self.n_mfcc
                        :frame_size + self.n_mfcc + self.n_mod_ceps]
        filtered_formant = x[:, frame_size + self.n_mfcc + self.n_mod_ceps:]
        # formant model
        formant_encoded = self.formant_encoder(filtered_formant.view(
                 -1, 1, frame_size)).view(n_samples, -1)
        formant_reconstructed = self.formant_decoder(
            self.formant_transform(
                torch.cat((formant_encoded, mfcc),
                          1)).view(n_samples, 5, -1)
        )

        return torch.cat(
            (
             formant_reconstructed.view(-1, frame_size),
             self.fundamental_freq_model(
                 torch.cat((fundamental_freq_dat, mfcc), 1))
            ), 1)
