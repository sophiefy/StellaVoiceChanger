from munch import Munch
import torch
from models_starganv2 import Generator, MappingNetwork, StyleEncoder
import yaml
from backend.starganv2.model_jdc import JDCNet

def build_models(hps):
    args = Munch(hps)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf,
                          F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains,
                                     hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)

    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

if __name__ == '__main__':
    device = torch.device('cpu')  # TODO

    config_path = "D:\Desktop\DUMMY\models\starganv2-1\config_starganv2.yml"
    model_path = "D:\Desktop\DUMMY\models\stargan.pth"

    with open(config_path) as f:
        starganv2_config = yaml.safe_load(f)

    starganv2 = build_models(starganv2_config["model_params"])

    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']
    print(params['mapping_network'].keys())
    _ = [starganv2[key].load_state_dict(params[key], False) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to(device)
    starganv2.mapping_network = starganv2.mapping_network.to(device)
    starganv2.generator = starganv2.generator.to(device)
