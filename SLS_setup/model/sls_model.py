import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    #Wav2Vec2BertConfig,
    #Wav2Vec2BertModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
)


HUB_NAME = "facebook/w2v-bert-2.0"
#HUB_NAME = "nii-yamagishilab/xls-r-2b-anti-deepfake"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def ssl_path(name=HUB_NAME):
    return name


############################
## FRONT-END: XLS-R or W2V-BERT 2.0
############################


class SSLModel(nn.Module):

    def __init__(self, device, n_layers=None, name=HUB_NAME,
                 devices=None, layer_split=None, out_dim=None):
        super().__init__()
        name = ssl_path(name)
        self.is_w2v_bert = ("w2v-bert" in name.lower())

        if self.is_w2v_bert:
            cfg = Wav2Vec2BertConfig.from_pretrained(name)
            cfg.layerdrop = 0.0
            cfg.apply_spec_augment = False
            

            if n_layers is None:
                n_layers = cfg.num_hidden_layers

            try:
                self.model = Wav2Vec2BertModel.from_pretrained(name, config=cfg, dtype=torch.float32)
            except TypeError:
                self.model = Wav2Vec2BertModel.from_pretrained(name, config=cfg, torch_dtype=torch.float32)
                
            self.model_dim = cfg.hidden_size
        else:
            cfg = Wav2Vec2Config.from_pretrained(name)
            cfg.layerdrop = 0.0
            cfg.apply_spec_augment = False
            cfg.mask_time_prob = 0.0
            cfg.mask_feature_prob = 0.0

            if n_layers is None:
                n_layers = 48

            try:
                self.model = Wav2Vec2Model.from_pretrained(name, config=cfg, dtype=torch.float32)
            except TypeError:
                self.model = Wav2Vec2Model.from_pretrained(name, config=cfg, torch_dtype=torch.float32)
                
            self.model_dim = 1920 if "xls-r-2b" in name else cfg.hidden_size

        max_layers = cfg.num_hidden_layers
        if n_layers > max_layers:
            print(f"[SSLModel] {name} has only {max_layers} layers; n_layers={n_layers} clamped to {max_layers}")
            n_layers = max_layers
        self.model.encoder.layers = self.model.encoder.layers[:n_layers]
        self.model.config.num_hidden_layers = n_layers

        self.name = name
        self.device = device
        self.n_layers = n_layers
        self.out_dim = out_dim if out_dim is not None else self.model_dim

        self.devices = None
        self.layer_counts = None
        if devices is None:
            self.to(device)
        else:
            self.split(devices, layer_split)

    def split(self, devices, layer_split=None):
        devices = [torch.device(f"cuda:{d}") for d in devices]
        layers = self.model.encoder.layers
        if layer_split is None:
            base, rem = divmod(len(layers), len(devices))
            layer_split = [base + (1 if i >= len(devices) - rem else 0) for i in range(len(devices))]
        assert len(layer_split) == len(devices) and sum(layer_split) == len(layers), \
            f"layer_split {layer_split} must have {len(devices)} entries summing to {len(layers)}"

        # everything that is not a transformer layer (CNN/feature projection, pos_conv_embed,
        # masked_spec_embed, W2V-BERT adapter/intermediate_ffn, ...) starts on the first card;
        # the layers and the final encoder norm are then moved to their own cards below
        self.model.to(devices[0])
        if getattr(self.model.encoder, "layer_norm", None) is not None:
            self.model.encoder.layer_norm.to(devices[-1])

        start = 0
        for dev, n in zip(devices, layer_split):
            for layer in layers[start:start + n]:
                layer.to(dev)
            if start > 0:
                layers[start].register_forward_pre_hook(self._relay_to(dev), with_kwargs=True)
            start += n

        self.devices = devices
        self.layer_counts = layer_split
        self.device = devices[0]

    @staticmethod
    def _relay_to(dev):
        def hook(module, args, kwargs):
            args = tuple(a.to(dev) if torch.is_tensor(a) else a for a in args)
            kwargs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            return args, kwargs
        return hook

    @property
    def output_device(self):
        return self.devices[-1] if self.devices else self.device

    def n_frames(self, n_samples):
        if not self.is_w2v_bert:
            return int(self.model._get_feat_extract_output_lengths(torch.tensor(n_samples)))
        # SeamlessM4TFeatureExtractor: 25 ms window / 10 ms hop, no centering, then frames are
        # padded to a multiple of stride=2 and stacked in pairs -> 160-dim features
        n_mel = 1 + (n_samples - 400) // 160
        n_frames = -(-n_mel // 2)
        if self.model.config.add_adapter:
            cfg = self.model.config
            for _ in range(cfg.num_adapter_layers):
                n_frames = (n_frames + 2 * (cfg.adapter_kernel_size // 2) - cfg.adapter_kernel_size) \
                    // cfg.adapter_stride + 1
        return int(n_frames)

    def forward(self, x):
        x = x.to(self.device)
        
        if self.is_w2v_bert:
            outputs = self.model(input_features=x, output_hidden_states=True)
        else:
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-7)
            outputs = self.model(x, output_hidden_states=True)
            
        hs = outputs.hidden_states[1:]
        hs = [h.to(self.output_device) for h in hs]
        return torch.stack(hs, dim=1)



class SLS(nn.Module):
    """Sensitive Layer Selection: sigmoid weight per layer, weighted sum of layers."""

    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, H):
        alpha = torch.sigmoid(self.fc(H.mean(dim=2)))        # (bs, layer, 1)
        fused = (H * alpha.unsqueeze(-1)).sum(dim=1)          # (bs, frame, dim)
        return fused


class ModelSLS(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        name = getattr(args, "ssl_name", HUB_NAME)
        default_layers = 24 if "w2v-bert" in name.lower() else 48
        n_layers = getattr(args, "n_layers", default_layers)
        devices = getattr(args, "devices", None)
        layer_split = getattr(args, "layer_split", None)

        self.ssl_model = SSLModel(
            device,
            n_layers=n_layers,
            name=name,
            devices=devices,
            layer_split=layer_split
        )
        self.sls = SLS(self.ssl_model.out_dim)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        
        n_frames = self.ssl_model.n_frames(64600)             
        self.fc1 = nn.Linear((n_frames // 3) * (self.ssl_model.out_dim // 3), 1024)  
        self.fc2 = nn.Linear(1024, 2)

        for m in (self.sls, self.first_bn, self.fc1, self.fc2):
            m.to(self.ssl_model.output_device)

    @property
    def input_device(self):
        return self.ssl_model.device

    @property
    def output_device(self):
        return self.ssl_model.output_device

    def forward(self, x):

        if x.dim() == 3 and not self.ssl_model.is_w2v_bert:
            x = x.squeeze(-1)
            
        H = self.ssl_model(x)                   # (bs, layers, frames, out_dim)
        x = self.sls(H)                         # (bs, frames, out_dim)

        x = x.unsqueeze(dim=1)                  
        x = self.first_bn(x)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.selu(self.fc1(x))
        output = self.fc2(x)
        return output