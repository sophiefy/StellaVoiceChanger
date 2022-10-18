import sys
import os
import torch
import librosa
from scipy.io.wavfile import write
import logging
import utils
from models import SynthesizerTrn
from hubert import load_hubert

# global variable
source_path = ''
flag_upload = False # 等待输入source path
flag_convert = False # 是否等待启动convert

class Sovits():
    def __init__(self, hubert_path, vits_path, hps):
        self.hubert_path = hubert_path
        self.vits_path = vits_path
        self.save_path = '../sovits_cache/temp.wav'
        self.hps = hps

        self._load_models()

    def _load_models(self):
        try:
            hubert = load_hubert(self.hubert_path)
        except:
            print('Failed to load hubert model!')

        vits = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model)
        _ = vits.eval()

        try:
            _ = utils.load_checkpoint(self.vits_path, vits)
        except:
            print('Failed to load vits model!')

        self.hubert = hubert
        self.vits = vits

    def inferene(self, source, noise_scale=0.667, noise_scale_w=0.8, length_scale=0.6):
        with torch.inference_mode():
            # extract speech units
            unit = self.hubert.units(source)
            unit = torch.FloatTensor(unit)
            unit_lengths = torch.LongTensor([unit.size(1)])
            # eonvert voice
            converted = self.vits.infer(
                unit,
                unit_lengths,
                noise_scale=.667,
                noise_scale_w=0.8,
                length_scale=0.6)[0][0,0].data.float().numpy()

        # write converted audio to cache
        write(self.save_path, 22050, converted)

def get_logger(filename='test.log'):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(lineno)s %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def ask_if_continue():
    while True:
        res = input('Contiue? (y/n): ')
        if res == 'y':
            break
        elif res == 'n':
            sys.exit(0)

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

def revise_path(origin_path):
    origin_path = origin_path.replace('\\', '/').replace('\n', '/n').replace('\r', '/r')
    converted_path = origin_path.replace('\t', '/t').replace('\a', '/a').replace('\b', 'b')

    return converted_path

def voice_conversion(hubert_path, vits_path, config_path):
    print('Loading models...')
    hps = utils.get_hparams_from_file(config_path)
    sovits = Sovits(hubert_path, vits_path, hps)
    print('Successfully loaded models!')

    while True:
        global flag_upload, flag_convert, source_path
        flag_upload = flag_convert= False

        print('Please input some audio...')
        wait_upload()
        source_path = revise_path(source_path)
        print(f'Successfully loaded audio from {source_path}')

        file_name = os.path.split(source_path)

        source, sr = librosa.load(source_path)
        source = librosa.resample(source, sr, 22050)
        source = librosa.to_mono(source)
        source = torch.from_numpy(source).unsqueeze(0).unsqueeze(1)

        wait_convert()
        print('Converting...')
        sovits.inferene(source)
        print('Successfully converted the source audio!')



