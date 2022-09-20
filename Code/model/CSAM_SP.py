import scipy
import math
import time
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from scipy import signal
from denoiser.resample import downsample2, upsample2
from denoiser.utils import capture_init


# CSynthesizerAttention************************************************************************************************************
class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(th.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = th.arange(length_q)
        range_vec_k = th.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = th.clamp(distance_mat, -self.max_relative_position, 0)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = th.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class SynthesizerAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        """

        n_embd : embedding size
        n_head : number of attention heads
        block_size : length of seq
        attn_pdrop : attention dropout probability
        resid_pdrop : dropout prob after projection layer.

        """
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.head_dim = self.n_embd // n_head
        self.w1 = nn.Linear(n_embd, n_embd)
        self.w2 = nn.Parameter(th.zeros(self.head_dim, self.n_embd*2))  # d_k,T
        self.b2 = nn.Parameter(th.zeros(self.n_embd*2))  # T
        self.value = nn.Linear(self.n_embd, self.n_embd)  # dmodel,dmodel
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(self.n_embd, self.n_embd)  # dmodel,dmodel
        self.n_head = n_head

        nn.init.uniform_(self.w2, -0.001, 0.001)


        self.max_relative_position = 5
        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # 1 499 768
        d_k = C // self.n_head
        relu_out = F.relu(self.w1(x)).view(B, T, self.n_head, d_k).transpose(1, 2)  # 32 8 249 96(QK)*********96 768
        scores = (relu_out @ self.w2) + self.b2  # 1 8 499 768(V)
        scores = scores[:, :, :T, :T]
        mask = th.tril(th.ones(T, T)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, -1e10)
        prob_attn = F.softmax(scores, dim=-1)  # 1 8 499 499

        v = self.value(x).view(B, T, self.n_head, d_k).transpose(1, 2)  # 1 8 499 96
        y1 = th.matmul(prob_attn, v)  # 1 8 499 96
        v2 = self.relative_position_v(T, T)  # 499 499 96
        y2 = prob_attn.permute(2, 0, 1, 3).contiguous().view(T, B * self.n_head, T)  # 499 8 499
        y2 = th.matmul(y2, v2)  # 499 8 96
        y2 = y2.transpose(0, 1).contiguous().view(B, self.n_head, T, self.head_dim)
        y = y1 + y2  # 1 8 499 96
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 1 499 768
        y = self.resid_drop(self.proj(y))  # 1 499 768
        return y


# ************************************************************************************************************

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


class CSAM_SP(nn.Module):
    """
    CSAM_SP speech enhancement model.
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

        if rescale:
            rescale_module(self, reference=rescale)
        # ************************************************************************************
        self.pos_atten = SynthesizerAttention(n_embd=768, n_head=8, attn_pdrop=0.1, resid_pdrop=0.1)
        # ************************************************************************************

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
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
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(0, 2, 1)
        x = self.pos_atten(x)
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


class CSAM_SPStreamer:
    """
    Streaming implementation for CSAM_SP. It supports being fed with any amount
    of audio at a time. You will get back as much audio as possible at that
    point.
    Args:
        - csam_sp (CSAM_SP): CSAM_SP model.
        - dry (float): amount of dry (e.g. input) signal to keep. 0 is maximum
            noise removal, 1 just returns the input signal. Small values > 0
            allows to limit distortions.
        - num_frames (int): number of frames to process at once. Higher values
            will increase overall latency but improve the real time factor.
        - resample_lookahead (int): extra lookahead used for the resampling.
        - resample_buffer (int): size of the buffer of previous inputs/outputs
            kept for resampling.
    """

    def __init__(self, csam_sp,
                 dry=0,
                 num_frames=1,
                 resample_lookahead=64,
                 resample_buffer=256):
        device = next(iter(csam_sp.parameters())).device
        self.csam_sp = csam_sp
        self.lstm_state = None
        self.conv_state = None
        self.dry = dry
        self.resample_lookahead = resample_lookahead
        resample_buffer = min(csam_sp.total_stride, resample_buffer)
        self.resample_buffer = resample_buffer
        self.frame_length = csam_sp.valid_length(1) + csam_sp.total_stride * (num_frames - 1)
        self.total_length = self.frame_length + self.resample_lookahead
        self.stride = csam_sp.total_stride * num_frames
        self.resample_in = th.zeros(csam_sp.chin, resample_buffer, device=device)
        self.resample_out = th.zeros(csam_sp.chin, resample_buffer, device=device)

        self.frames = 0
        self.total_time = 0
        self.variance = 0
        self.pending = th.zeros(csam_sp.chin, 0, device=device)

        bias = csam_sp.decoder[0][2].bias
        weight = csam_sp.decoder[0][2].weight
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
        padding = th.zeros(self.csam_sp.chin, self.total_length, device=self.pending.device)
        out = self.feed(padding)
        return out[:, :pending_length]

    def feed(self, wav):
        """
        Apply the model to mix using true real time evaluation.
        Normalization is done online as is the resampling.
        """
        begin = time.time()
        csam_sp = self.csam_sp
        resample_buffer = self.resample_buffer
        stride = self.stride
        resample = csam_sp.resample

        if wav.dim() != 2:
            raise ValueError("input wav should be two dimensional.")
        chin, _ = wav.shape
        if chin != csam_sp.chin:
            raise ValueError(f"Expected {csam_sp.chin} channels, got {chin}")

        self.pending = th.cat([self.pending, wav], dim=1)
        outs = []
        while self.pending.shape[1] >= self.total_length:
            self.frames += 1
            frame = self.pending[:, :self.total_length]
            dry_signal = frame[:, :stride]
            if csam_sp.normalize:
                mono = frame.mean(0)
                variance = (mono ** 2).mean()
                self.variance = variance / self.frames + (1 - 1 / self.frames) * self.variance
                frame = frame / (csam_sp.floor + math.sqrt(self.variance))
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

            if csam_sp.normalize:
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
        csam_sp = self.csam_sp
        skips = []
        next_state = []
        first = self.conv_state is None
        stride = self.stride * csam_sp.resample
        x = frame[None]
        for idx, encode in enumerate(csam_sp.encoder):
            stride //= csam_sp.stride
            length = x.shape[2]
            if idx == csam_sp.depth - 1:
                # This is sligthly faster for the last conv
                x = fast_conv(encode[0], x)
                x = encode[1](x)
                x = fast_conv(encode[2], x)
                x = encode[3](x)
            else:
                if not first:
                    prev = self.conv_state.pop(0)
                    prev = prev[..., stride:]
                    tgt = (length - csam_sp.kernel_size) // csam_sp.stride + 1
                    missing = tgt - prev.shape[-1]
                    offset = length - csam_sp.kernel_size - csam_sp.stride * (missing - 1)
                    x = x[..., offset:]
                x = encode[1](encode[0](x))
                x = fast_conv(encode[2], x)
                x = encode[3](x)
                if not first:
                    x = th.cat([prev, x], -1)
                next_state.append(x)
            skips.append(x)


        x = x.permute(0, 2, 1)
        x = csam_sp.pos_atten(x)
        x = x.permute(0, 2, 1)

        extra = None
        for idx, decode in enumerate(csam_sp.decoder):
            skip = skips.pop(-1)
            x += skip[..., :x.shape[-1]]
            x = fast_conv(decode[0], x)
            x = decode[1](x)

            if extra is not None:
                skip = skip[..., x.shape[-1]:]
                extra += skip[..., :extra.shape[-1]]
                extra = decode[2](decode[1](decode[0](extra)))
            x = decode[2](x)
            next_state.append(x[..., -csam_sp.stride:] - decode[2].bias.view(-1, 1))
            if extra is None:
                extra = x[..., -csam_sp.stride:]
            else:
                extra[..., :csam_sp.stride] += next_state[-1]
            x = x[..., :-csam_sp.stride]

            if not first:
                prev = self.conv_state.pop(0)
                x[..., :csam_sp.stride] += prev
            if idx != csam_sp.depth - 1:
                x = decode[3](x)
                extra = decode[3](extra)
        self.conv_state = next_state
        return x[0], extra[0]


def test():
    import argparse
    parser = argparse.ArgumentParser(
        "denoiser.csam_sp",
        description="Benchmark the streaming csam_sp implementation, "
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
    csam_sp = CSAM_SP(depth=args.depth, hidden=args.hidden, resample=args.resample).to(args.device)
    x = th.randn(1, int(sr * 8)).to(args.device)
    out = csam_sp(x[None])[0]
    streamer = CSAM_SPStreamer(csam_sp, num_frames=args.num_frames)
    out_rt = []
    frame_size = streamer.total_length
    with th.no_grad():
        while x.shape[1] > 0:
            out_rt.append(streamer.feed(x[:, :frame_size]))
            x = x[:, frame_size:]
            frame_size = streamer.csam_sp.total_stride
    out_rt.append(streamer.flush())
    out_rt = th.cat(out_rt, 1)
    model_size = sum(p.numel() for p in csam_sp.parameters()) * 4 / 2 ** 20
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
