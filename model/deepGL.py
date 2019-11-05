import torch
import torch.nn as nn
import os
import numpy as np
import time
from torch.optim import Adam
from model.istft import InverseSTFT
from model.display import stream
from frontend.audio_preprocess import _istft, save_dgl_wav


def replace_magnitude(x, mag):
    phase = torch.atan2(x[:, 1:], x[:, :1])
    return torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)


class DeGLI(nn.Module):
    def __init__(self, n_fft, hop_length, win_size, ch_hidden, depth, out_all_block=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_size = win_size
        self.out_all_block = out_all_block

        #self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        self.istft = InverseSTFT(n_fft, hop_length=hop_length, win_length=win_size)

        self.dnns = nn.ModuleList([DeGLI_DNN(ch_hidden) for _ in range(depth)])

        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.num_params()

    def forward(self, x, mag, max_length=None, repeat=1):
        max_len = max_length
        if isinstance(max_length, torch.Tensor):
            max_len = int(max_length.item())

        out_repeats = []
        for ii in range(repeat):
            for dnn in self.dnns:
                mag_replaced = replace_magnitude(x, mag)
                waves = self.istft(mag_replaced.permute(0, 2, 3, 1), length=max_len)
                consistent = self.stft(waves)
                consistent = consistent.permute(0, 3, 1, 2)
                residual = dnn(x, mag_replaced, consistent)
                x = consistent - residual

            if self.out_all_block:
                out_repeats.append(x)

        if self.out_all_block:
            out_repeats = torch.stack(out_repeats, dim=1)
        else:
            out_repeats = x.unsqueeze(1)

        final_out = replace_magnitude(x, mag)

        return  out_repeats, final_out, residual

    def stft(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_size)

    def get_step(self):
        return self.step.data.item()

    def checkpoint(self, path):
        k_steps = self.get_step() // 1000
        self.save(f'{path}/checkpoint_{k_steps}k_steps.pyt')

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def restore(self, path):
        if not os.path.exists(path):
            print('\nNew DeepGL Training Session...\n')
            self.save(path)
        else:
            print(f'\nLoading Weights: "{path}"\n')
            self.load(path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)

class DeGLI_DNN(nn.Module):
    def __init__(self, ch_hidden=16):
        super().__init__()
        self.convglu_first = ConvGLU(6, ch_hidden, kernel_size=(11, 11), batchnorm=True)
        self.two_convglus = nn.Sequential(
            ConvGLU(ch_hidden, ch_hidden, batchnorm=True),
            ConvGLU(ch_hidden, ch_hidden, batchnorm=True)
        )
        self.convglu_last = ConvGLU(ch_hidden, ch_hidden)
        self.conv = nn.Conv2d(ch_hidden, 2, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, x, mag_replaced, consistent):
        x = torch.cat([x, mag_replaced, consistent], dim=1)
        x = self.convglu_first(x)
        residual = x
        x = self.two_convglus(x)
        x += residual
        x = self.convglu_last(x)
        x = self.conv(x)
        return x


class ConvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(7, 7), padding=None, batchnorm=False):
        super().__init__()
        if not padding:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=padding)
        if batchnorm:
            self.conv = nn.Sequential(
                self.conv,
                nn.BatchNorm2d(out_ch * 2)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        ch = x.shape[1]
        x = x[:, :ch//2, ...] * self.sigmoid(x[:, ch//2:, ...])
        return x


class Opt_DeGLI:
    def __init__(self, hp, path, lr, loss):
        self.model = DeGLI(hp.n_fft,
                           hp.hop_length,
                           hp.win_size,
                           hp.ch_hidden,
                           hp.depth,
                           hp.out_all_block).cuda()
        self._hp = hp
        self.path = path
        self.loss = loss
        self.optim = Adam(self.model.parameters(),
                          lr=lr,
                          weight_decay=hp.weight_decay,
                         )

    def train(self, train_set, test_set, total_steps):

        total_iters = len(train_set)
        epochs = (total_steps - self.model.get_step()) // total_iters + 1

        for e in range(1, epochs + 1):

            start = time.time()
            running_loss = 0.

            for i, (B_x, B_y, B_mag, B_f, B_len) in enumerate(train_set, 1):

                x, y, m = B_x.cuda(), B_y.cuda(), B_mag.cuda()
                max_len = max(B_len).cuda()

                frames = B_f.cuda()

                output_loss, _, _ = self.model(x, m, max_len,
                                               repeat=self._hp.repeat_train)

                loss = self.calc_loss(output_loss, y, frames)

                self.optim.zero_grad()
                loss.backward()
                # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                #                                            self._hp.thr_clip_grad)
                self.optim.step()
                running_loss += loss

                speed = i / (time.time() - start)
                avg_loss = running_loss / i

                step = self.model.get_step()
                k = step // 1000

                if step % self._hp.voc_checkpoint_every == 0:
                    self.gerator(test_set, k)

                    self.model.checkpoint(self.path.voc_checkpoints)

                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
                stream(msg)

            self.model.save(self.path.voc_latest_weights)
            self.model.log(self.path.voc_log, msg)
            print(' ')

    @torch.no_grad()
    def gerator(self, test_data, k):
        for i, (t_x, t_y, t_mag, t_f, t_len) in enumerate(test_data, 1):
            t_x, t_m = t_x.cuda(), t_mag.cuda()
            max_len = max(t_len).cuda()
            output_loss, output, residual = self.model(t_x, t_m, max_len,
                                                       repeat=self._hp.repeat_train)
            dict_one = self.postprocess(output, residual)
            mag = dict_one['out'].squeeze()
            phase = dict_one['res'].squeeze()
            spec = mag * np.exp(1j * phase)
            wave = _istft(spec, self._hp)
            save_dgl_wav(wave, f'{self.path.voc_output}{k}k_steps_{i}_target.wav', self._hp.sample_rate)

    @torch.no_grad()
    def postprocess(self, output, residual):
        dict_one = dict(out=output, res=residual)
        for key in dict_one:
            one = dict_one[key][0, :, :, :]
            one = one.permute(1, 2, 0).contiguous()  # F, T, 2

            one = one.cpu().numpy().view(dtype=np.complex64)  # F, T, 1
            dict_one[key] = one

        return dict_one

    def calc_loss(self, output_loss, y, frames):
        loss_no_red =self.loss(output_loss, y.unsqueeze(1))
        loss_blocks = torch.zeros(output_loss.shape[1]).cuda()

        for T, loss_batch in zip(frames, loss_no_red):
            T = int(T)
            loss_blocks += torch.mean(loss_batch[..., :T], dim=(1, 2, 3))

        loss_weight = torch.Tensor(
            [1. / i for i in range(self._hp.repeat_train, 0, -1)]).cuda()
        loss_weight /= loss_weight.sum()

        if len(loss_blocks) == 1:
            loss = loss_blocks.squeeze()
        else:
            loss = loss_blocks @ loss_weight

        return loss

    def restore(self, checkpoint):
        self.model.restore(checkpoint)

    def get_step(self):
        return self.model.get_step()


