import os
import torch
import librosa
from scipy.io.wavfile import write
from pydub import AudioSegment
import utils
import shutil
from mel_processing import spectrogram_torch
from models import SynthesizerTrn
from hubert import load_hubert

# global variable
source_path = ''
export_path = 'cache/wavefile.wav'
flag_upload = False  # 等待输入source path
flag_convert = False  # 是否等待启动convert
flag_terminate = False  # 是否结束线程
flag_mode = False  # vc模式。False为hubert，True为flow
sid_source = None  # 实际使用时，只有sid_target会用上
sid_target = None
flag_device = False  # CPU为False，GPU为True
device = torch.device('cpu')  # TODO
NOISE_SCALE = 0.667
NOISE_SCALE_W = 0.8
LENGTH_SCALE = 0.6


class Sovits():
    def __init__(self,
                 hubert_path,
                 vits_path,
                 hps,
                 device,
                 save_path='cache/converted.wav',
                 play_path='cache/play.wav',
                 download_path='cache/download.wav'):

        self.hubert_path = hubert_path
        self.vits_path = vits_path
        self.save_path = save_path
        self.play_path = play_path
        self.download_path = download_path
        self.hps = hps
        self.device = device

        self.load_models()

    def load_models(self):
        try:
            hubert = load_hubert(self.hubert_path)
            self.hubert = hubert.to(self.device)
        except:
            raise FileNotFoundError('Failed to load hubert model!')

        vits = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model)
        _ = vits.eval()

        try:
            _ = utils.load_checkpoint(self.vits_path, vits)
            self.vits = vits.to(self.device)
        except:
            raise FileNotFoundError('Failed to load vits model!')

    def inference_hubert(self, source, sid=None, noise_scale=0.667, noise_scale_w=0.8, length_scale=0.6):
        source = source.to(device)

        with torch.inference_mode():
            # extract speech units
            unit = self.hubert.units(source)
            unit = torch.FloatTensor(unit.cpu())
            unit = unit.to(device)
            unit_lengths = torch.LongTensor([unit.size(1)]).to(device)
            # convert voice
            # single speaker
            if sid is None:
                converted = self.vits.infer(
                    unit,
                    unit_lengths,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale)[0][0, 0].data.float().cpu().numpy()
            # multi-speaker
            else:
                converted = self.vits.infer(
                    unit,
                    unit_lengths,
                    sid,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale)[0][0, 0].data.float().cpu().numpy()

        # write converted audio to cache
        write(self.save_path, 22050, converted)
        shutil.copy(self.save_path, self.play_path)
        shutil.copy(self.save_path, self.download_path)

    def inference_flow(self, source, sid_src, sid_tgt):
        spec = spectrogram_torch(source, self.hps.data.filter_length,
                                 self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                 center=False)
        spec_lengths = torch.LongTensor([spec.size(-1)])

        with torch.inference_mode():
            converted = self.vits.voice_conversion(
                spec, spec_lengths,
                sid_src=sid_src,
                sid_tgt=sid_tgt)[0][0, 0].data.float().numpy()

        # write converted audio to cache
        write(self.save_path, 22050, converted)
        shutil.copy(self.save_path, self.play_path)
        shutil.copy(self.save_path, self.download_path)


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


# hubert or flow
def change_mode():
    global flag_mode
    if flag_mode:
        flag_mode = False
    else:
        flag_mode = True


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

    source_hubert, sr = librosa.load(source_path)
    source_hubert = librosa.resample(source_hubert, sr, 22050)
    source_hubert = librosa.to_mono(source_hubert)
    source_hubert = torch.from_numpy(source_hubert).unsqueeze(0).unsqueeze(1)

    source_flow, _ = utils.load_wav_to_torch(source_path)
    source_flow = source_flow / 32768.0
    source_flow = source_flow.unsqueeze(0)

    return source_hubert, source_flow


def select_speaker(speaker_id):
    global sid_target
    sid_target = torch.LongTensor([speaker_id])


def terminate_vc():
    global flag_upload, flag_terminate
    flag_terminate = True
    flag_upload = True


def voice_conversion(model_dir):
    hubert_path = os.path.join(model_dir, 'hubert.pt')
    vits_path = os.path.join(model_dir, 'vits.pth')
    config_path = os.path.join(model_dir, 'config.json')
    print('Loading models...')
    hps = utils.get_hparams_from_file(config_path)
    sovits = Sovits(hubert_path, vits_path, hps, device)
    print('Successfully loaded models!')

    while True:
        global flag_upload, flag_convert, flag_terminate, source_path, sid_source, sid_target
        flag_upload = flag_convert = flag_terminate = False

        print('Please input some audio...')
        wait_upload()

        if flag_terminate:
            print('Terminating...')
            break

        source_path = revise_path(source_path)
        print(f'Successfully loaded audio from {source_path}')

        source_hubert, source_flow = load_wav(source_path)

        wait_convert()

        if not flag_mode:
            print('Stella Voice Changer> Converting (hubert mode)...')
            sovits.inference_hubert(source_hubert, sid_target)
        else:
            print('Converting (flow mode)...')
            # vits的vc不是any-to-many，但可以通过自己转自己的方式提升效果，即认为source也是target
            sovits.inference_flow(source_flow, sid_target, sid_target)
        print('Successfully converted the source audio!')

    print('Terminated!')

# if __name__ == "__main__":
# debug use
# hubert_path = '../backend/hubert-soft.pt'
# vits_path = '../backend/sovits-nat-1.pth'
# config_path = '../backend/config-sovits-nat-1.json'
# voice_conversion(hubert_path, vits_path, config_path)
