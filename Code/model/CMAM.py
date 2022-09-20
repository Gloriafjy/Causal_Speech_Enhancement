import math
import time
import torch as th
from torch import nn
from torch.nn import functional as F
from denoiser.resample import downsample2, upsample2
from denoiser.utils import capture_init


class CausalSelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 5

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = th.sqrt(th.FloatTensor([self.head_dim]))

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = th.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        mask = th.tril(th.ones(len_q, len_q))
        attn = attn.masked_fill(mask == 0, -1e10)

        # # 888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
        # # scores1 = scores[:, :, :T, :T]
        # attn = attn.masked_fill(mask == 0, 0)
        # #
        # plt.matshow(attn[0, 0, :, :].cpu().detach().numpy())
        # plt.matshow(attn[0, 1, :, :].cpu().detach().numpy())
        # plt.matshow(attn[0, 2, :, :].cpu().detach().numpy())
        # plt.matshow(attn[0, 3, :, :].cpu().detach().numpy())
        # plt.matshow(attn[0, 4, :, :].cpu().detach().numpy())
        # plt.matshow(attn[0, 5, :, :].cpu().detach().numpy())
        # plt.matshow(attn[0, 6, :, :].cpu().detach().numpy())
        # plt.matshow(attn[0, 7, :, :].cpu().detach().numpy())
        # plt.show()
        # print('a')
        # # 888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

        attn = self.dropout(th.softmax(attn, dim=-1))

        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        x = th.matmul(attn, r_v1)  # 32 8 249 96
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class CMAM(nn.Module):
    """
    CMAM speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.
    """

    @capture_init
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        # CausalSelfAttention*******************************************************************************************************
        self.causalattention = CausalSelfAttention(hid_dim=768, n_heads=8, dropout=0.1, device="cuda")
        # *******************************************************************************************************

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)  # 64000*4=256000
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1  # 256000 63999 15999 3999 999 249
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size  # 249 1000 4004 16020 64084 256340
        length = int(math.ceil(length / self.resample))  # 64085
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]  # 64085
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))  # 16 1 64085
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)  # 16 1 128170
            x = upsample2(x)  # 16 1 256340
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(0, 2, 1)  # 16 768(d_model) 249(seq_len) -> 16 249 768
        x = self.causalattention(x, x, x)
        x = x.permute(0, 2, 1)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x


def fast_conv(conv, x):
    """
    Faster convolution evaluation if either kernel size is 1
    or length of sequence is 1.
    """
    batch, chin, length = x.shape
    chout, chin, kernel = conv.weight.shape
    assert batch == 1
    if kernel == 1:
        x = x.view(chin, length)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin), x)
    elif length == kernel:
        x = x.view(chin * kernel, 1)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin * kernel), x)
    else:
        out = conv(x)
    return out.view(batch, chout, -1)


class CMAMStreamer:
    """
    Streaming implementation for CMAM. It supports being fed with any amount
    of audio at a time. You will get back as much audio as possible at that
    point.

    Args:
        - cmam (CMAM): CMAM model.
        - dry (float): amount of dry (e.g. input) signal to keep. 0 is maximum
            noise removal, 1 just returns the input signal. Small values > 0
            allows to limit distortions.
        - num_frames (int): number of frames to process at once. Higher values
            will increase overall latency but improve the real time factor.
        - resample_lookahead (int): extra lookahead used for the resampling.
        - resample_buffer (int): size of the buffer of previous inputs/outputs
            kept for resampling.
    """

    def __init__(self, cmam,
                 dry=0,
                 num_frames=1,
                 resample_lookahead=64,
                 resample_buffer=256):
        device = next(iter(cmam.parameters())).device
        self.cmam = cmam
        self.lstm_state = None
        self.conv_state = None
        self.dry = dry
        self.resample_lookahead = resample_lookahead  # 64
        resample_buffer = min(cmam.total_stride, resample_buffer)  # 256
        self.resample_buffer = resample_buffer  # 256
        self.frame_length = cmam.valid_length(1) + cmam.total_stride * (num_frames - 1)  # 597
        self.total_length = self.frame_length + self.resample_lookahead  # 661
        self.stride = cmam.total_stride * num_frames  # 256
        self.resample_in = th.zeros(cmam.chin, resample_buffer, device=device)  # 256
        self.resample_out = th.zeros(cmam.chin, resample_buffer, device=device)  # 256

        self.frames = 0
        self.total_time = 0
        self.variance = 0
        self.pending = th.zeros(cmam.chin, 0, device=device)

        bias = cmam.decoder[0][2].bias
        weight = cmam.decoder[0][2].weight
        chin, chout, kernel = weight.shape
        self._bias = bias.view(-1, 1).repeat(1, kernel).view(-1, 1)
        self._weight = weight.permute(1, 2, 0).contiguous()

    def reset_time_per_frame(self):
        self.total_time = 0
        self.frames = 0

    @property
    def time_per_frame(self):
        return self.total_time / self.frames

    def flush(self):
        """
        Flush remaining audio by padding it with zero. Call this
        when you have no more input and want to get back the last chunk of audio.
        """
        pending_length = self.pending.shape[1]
        padding = th.zeros(self.cmam.chin, self.total_length, device=self.pending.device)
        out = self.feed(padding)
        return out[:, :pending_length]

    def feed(self, wav):
        """
        Apply the model to mix using true real time evaluation.
        Normalization is done online as is the resampling.
        """
        begin = time.time()
        cmam = self.cmam
        resample_buffer = self.resample_buffer
        stride = self.stride
        resample = cmam.resample

        if wav.dim() != 2:
            raise ValueError("input wav should be two dimensional.")
        chin, _ = wav.shape
        if chin != cmam.chin:
            raise ValueError(f"Expected {cmam.chin} channels, got {chin}")

        self.pending = th.cat([self.pending, wav], dim=1)
        outs = []
        while self.pending.shape[1] >= self.total_length:
            self.frames += 1
            frame = self.pending[:, :self.total_length]
            dry_signal = frame[:, :stride]
            if cmam.normalize:
                mono = frame.mean(0)
                variance = (mono ** 2).mean()
                self.variance = variance / self.frames + (1 - 1 / self.frames) * self.variance
                frame = frame / (cmam.floor + math.sqrt(self.variance))
            padded_frame = th.cat([self.resample_in, frame], dim=-1)
            self.resample_in[:] = frame[:, stride - resample_buffer:stride]
            frame = padded_frame

            if resample == 4:
                frame = upsample2(upsample2(frame))
            elif resample == 2:
                frame = upsample2(frame)
            frame = frame[:, resample * resample_buffer:]  # remove pre sampling buffer
            frame = frame[:, :resample * self.frame_length]  # remove extra samples after window

            out, extra = self._separate_frame(frame)
            padded_out = th.cat([self.resample_out, out, extra], 1)
            self.resample_out[:] = out[:, -resample_buffer:]
            if resample == 4:
                out = downsample2(downsample2(padded_out))
            elif resample == 2:
                out = downsample2(padded_out)
            else:
                out = padded_out

            out = out[:, resample_buffer // resample:]
            out = out[:, :stride]

            if cmam.normalize:
                out *= math.sqrt(self.variance)
            out = self.dry * dry_signal + (1 - self.dry) * out
            outs.append(out)
            self.pending = self.pending[:, stride:]

        self.total_time += time.time() - begin
        if outs:
            out = th.cat(outs, 1)
        else:
            out = th.zeros(chin, 0, device=wav.device)
        return out

    def _separate_frame(self, frame):
        cmam = self.cmam
        skips = []
        next_state = []
        first = self.conv_state is None
        stride = self.stride * cmam.resample
        x = frame[None]
        for idx, encode in enumerate(cmam.encoder):
            stride //= cmam.stride
            length = x.shape[2]
            if idx == cmam.depth - 1:
                # This is sligthly faster for the last conv
                x = fast_conv(encode[0], x)
                x = encode[1](x)
                x = fast_conv(encode[2], x)
                x = encode[3](x)
            else:
                if not first:
                    prev = self.conv_state.pop(0)
                    prev = prev[..., stride:]
                    tgt = (length - cmam.kernel_size) // cmam.stride + 1
                    missing = tgt - prev.shape[-1]
                    offset = length - cmam.kernel_size - cmam.stride * (missing - 1)
                    x = x[..., offset:]
                x = encode[1](encode[0](x))
                x = fast_conv(encode[2], x)
                x = encode[3](x)
                if not first:
                    x = th.cat([prev, x], -1)
                next_state.append(x)
            skips.append(x)

        x = x.permute(0, 2, 1)
        x = cmam.causalattention(x, x, x)
        x = x.permute(0, 2, 1)
        extra = None
        for idx, decode in enumerate(cmam.decoder):
            skip = skips.pop(-1)
            x += skip[..., :x.shape[-1]]
            x = fast_conv(decode[0], x)
            x = decode[1](x)

            if extra is not None:
                skip = skip[..., x.shape[-1]:]
                extra += skip[..., :extra.shape[-1]]
                extra = decode[2](decode[1](decode[0](extra)))
            x = decode[2](x)
            next_state.append(x[..., -cmam.stride:] - decode[2].bias.view(-1, 1))
            if extra is None:
                extra = x[..., -cmam.stride:]
            else:
                extra[..., :cmam.stride] += next_state[-1]
            x = x[..., :-cmam.stride]

            if not first:
                prev = self.conv_state.pop(0)
                x[..., :cmam.stride] += prev
            if idx != cmam.depth - 1:
                x = decode[3](x)
                extra = decode[3](extra)
        self.conv_state = next_state
        return x[0], extra[0]


def test():
    import argparse
    parser = argparse.ArgumentParser(
        "denoiser.cmam",
        description="Benchmark the streaming cmam implementation, "
                    "as well as checking the delta with the offline implementation.")
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--resample", default=4, type=int)
    parser.add_argument("--hidden", default=48, type=int)
    parser.add_argument("--sample_rate", default=16000, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-t", "--num_threads", type=int)
    parser.add_argument("-f", "--num_frames", type=int, default=1)
    args = parser.parse_args()
    if args.num_threads:
        th.set_num_threads(args.num_threads)
    sr = args.sample_rate
    sr_ms = sr / 1000
    cmam = CMAM(depth=args.depth, hidden=args.hidden, resample=args.resample).to(args.device)
    x = th.randn(1, int(sr * 4)).to(args.device)
    out = cmam(x[None])[0]
    streamer = CMAMStreamer(cmam, num_frames=args.num_frames)
    out_rt = []
    frame_size = streamer.total_length  # 661
    with th.no_grad():
        while x.shape[1] > 0:
            out_rt.append(streamer.feed(x[:, :frame_size]))
            x = x[:, frame_size:]
            frame_size = streamer.cmam.total_stride
    out_rt.append(streamer.flush())
    out_rt = th.cat(out_rt, 1)
    model_size = sum(p.numel() for p in cmam.parameters()) * 4 / 2 ** 20
    initial_lag = streamer.total_length / sr_ms
    tpf = 1000 * streamer.time_per_frame
    print(f"model size: {model_size:.1f}MB, ", end='')
    print(f"delta batch/streaming: {th.norm(out - out_rt) / th.norm(out):.2%}")
    print(f"initial lag: {initial_lag:.1f}ms, ", end='')
    print(f"stride: {streamer.stride * args.num_frames / sr_ms:.1f}ms")
    print(f"time per frame: {tpf:.1f}ms, ", end='')
    print(f"RTF: {((1000 * streamer.time_per_frame) / (streamer.stride / sr_ms)):.2f}")
    print(f"Total lag with computation: {initial_lag + tpf:.1f}ms")


if __name__ == "__main__":
    test()
