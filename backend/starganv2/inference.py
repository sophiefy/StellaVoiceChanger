import os
import yaml
import numpy as np
import torch
import torchaudio
import shutil
import librosa
from munch import Munch
from backend.starganv2.model_jdc import JDCNet
from models_starganv2 import Generator, MappingNetwork
from scipy.io.wavfile import write
from pydub import AudioSegment
from parallel_wavegan.utils import load_model

# globel variables
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

source_path = ''
export_path = 'cache/wavefile.wav'  # mp3等格式音频转成wav后的文件
flag_upload = False  # 等待输入source path
flag_convert = False  # 是否等待启动convert
flag_terminate = False  # 是否结束线程
sid_target = torch.LongTensor([0])  # target speaker
flag_device = False  # CPU为False，GPU为True
device = torch.device('cpu')  # TODO


class StarGANv2:
    def __init__(self,
                 f0_path,
                 starganv2_path,
                 vocoder_path,
                 hps,
                 device,
                 save_path='cache/converted.wav',
                 play_path='cache/play.wav',
                 download_path='cache/download.wav'):

        self.f0_path = f0_path
        self.starganv2_path = starganv2_path
        self.vocoder_path = vocoder_path
        self.hps = hps
        self.device = device
        self.save_path = save_path
        self.play_path = play_path
        self.download_path = download_path

        self.load_models()

    def build_models(self):
        args = Munch(self.hps)
        generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf,
                              F0_channel=args.F0_channel)
        mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains,
                                         hidden_dim=args.max_conv_dim)

        nets_ema = Munch(generator=generator,
                         mapping_network=mapping_network)

        return nets_ema

    def load_models(self):
        try:
            f0_model = JDCNet(num_class=1, seq_len=192)
            params = torch.load(self.f0_path, map_location='cpu')['net']
            f0_model.load_state_dict(params)
            _ = f0_model.eval()
            f0_model = f0_model.to(self.device)
            self.f0_model = f0_model
        except:
            raise FileNotFoundError('Failed to load f0 model!')

        try:
            vocoder = load_model(self.vocoder_path).to(self.device).eval()
            vocoder.remove_weight_norm()
            _ = vocoder.eval()
            self.vocoder = vocoder
        except:
            raise FileNotFoundError('Failed to load vocoder!')

        try:
            # TODO: 删除style encoder的权重
            starganv2 = self.build_models()
            params = torch.load(self.starganv2_path, map_location='cpu')
            _ = [starganv2[key].load_state_dict(params[key], False) for key in starganv2]
            _ = [starganv2[key].eval() for key in starganv2]
            starganv2.mapping_network = starganv2.mapping_network.to(device)
            starganv2.generator = starganv2.generator.to(device)
            self.mapping_network = starganv2.mapping_network
            self.generator = starganv2.generator
        except:
            raise FileNotFoundError('Failed to load starganv2!')

    def compute_style(self, sid):
        label = torch.LongTensor([sid]).to(device)
        latent_dim = self.mapping_network.shared[0].in_features
        ref = self.mapping_network(torch.randn(1, latent_dim).to(device), label)

        return ref

    def inference(self, source, sid):

        source = self.preprocess(source).to(self.device)

        ref = self.compute_style(sid)

        with torch.inference_mode():
            f0_feat = self.f0_model.get_feature_GAN(source.unsqueeze(1))
            out = self.generator(source.unsqueeze(1), ref, F0=f0_feat)
            c = out.transpose(-1, -2).squeeze().to(device)

            converted = self.vocoder.inference(c)
            converted = converted.view(-1).cpu().numpy()

        # write converted audio to cache
        write(self.save_path, 24000, converted)
        shutil.copy(self.save_path, self.play_path)
        shutil.copy(self.save_path, self.download_path)

    @staticmethod
    def preprocess(wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor


def wait_upload():
    global flag_upload
    while True:
        if flag_upload:
            break


def load_audio(audio_path):
    global source_path
    global flag_upload
    if audio_path:
        source_path = audio_path
        flag_upload = True


def wait_convert():
    global flag_convert
    while True:
        if flag_convert:
            break


def convert_audio():
    global flag_upload, flag_convert, source_path
    if source_path:
        flag_upload = True
        flag_convert = True


# TODO
def change_device():
    global flag_device, device
    if flag_device:
        flag_device = False
        device = torch.device('cpu')
    else:
        flag_device = True
        device = torch.device('cuda:0')


def revise_path(origin_path):
    origin_path = origin_path.replace('\\', '/').replace('\n', '/n').replace('\r', '/r')
    revised_path = origin_path.replace('\t', '/t').replace('\a', '/a').replace('\b', '/b')

    return revised_path


# 支持将常见音频格式转成wav
def load_wav(audio_path: str):
    global source_path, source_path
    if audio_path.endswith('wav'):
        pass
    else:
        if audio_path.endswith('mp3'):
            audio = AudioSegment.from_mp3(audio_path)

        elif audio_path.endswith('ogg'):
            audio = AudioSegment.from_ogg(audio_path)

        elif audio_path.endswith('flv'):
            audio = AudioSegment.from_flv(audio_path)
        else:
            raise ValueError('Not supported audio format!')

        audio.export(export_path, format='wav')
        source_path = export_path

    source, _ = librosa.load(source_path, sr=24000)
    source = source / np.max(np.abs(source))
    source.dtype = np.float32

    return source


def select_speaker(speaker_id):
    global sid_target
    sid_target = torch.LongTensor([speaker_id])


def terminate_vc():
    global flag_upload, flag_terminate
    flag_terminate = True
    flag_upload = True


def voice_conversion(model_dir):

    f0_path = os.path.join(model_dir, 'jdc.pth')
    starganv2_path = os.path.join(model_dir, 'starganv2.pth')
    vocoder_path = os.path.join(model_dir, 'vocoder.pkl')
    config_path = os.path.join(model_dir, 'config_starganv2.yml')

    print('Loading models...')
    with open(config_path) as f:
        hps = yaml.safe_load(f)
    hps = hps['model_params']
    starganv2 = StarGANv2(f0_path, starganv2_path, vocoder_path, hps, device)
    print('Successfully loaded models!')

    while True:
        global flag_upload, flag_convert, flag_terminate, source_path, sid_target
        flag_upload = flag_convert = flag_terminate = False

        print('Please input some audio...')
        wait_upload()

        if flag_terminate:
            print('Terminating...')
            break

        source_path = revise_path(source_path)
        print(f'Successfully loaded audio from {source_path}')

        source = load_wav(source_path)

        wait_convert()

        print('Converting...')
        starganv2.inference(source, sid_target)
        print('Successfully converted the source audio!')

    print('Terminated!')
