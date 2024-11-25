"""Helper code to instantiate various models."""

import torch
import torchvision

from collections import OrderedDict

from .resnets import ResNet, resnet_depths_to_config
from .densenets import DenseNet, densenet_depths_to_config
from .nfnets import NFNet
from .vgg import VGG

from .language_models import RNNModel, TransformerModel, LinearModel
from .losses import CausalLoss, MLMLoss, MostlyCausalLoss, CustomMSELoss, NewCrossEntropy, MixupCrossEntropy


def construct_model(cfg_model, cfg_data, model, pretrained=True, **kwargs):
    if cfg_data.modality == "vision":
        model = _construct_vision_model(cfg_model, cfg_data, model, pretrained, **kwargs)
    elif cfg_data.modality == "text":
        model = _construct_text_model(cfg_model, cfg_data, pretrained, **kwargs)
    else:
        raise ValueError(f"Invalid data modality {cfg_data.modality}")
    # Save nametag for printouts later:
    model.name = cfg_model

    # Choose loss function according to data and model:
    if "classification" in cfg_data.task:
        loss_fn = torch.nn.CrossEntropyLoss()
    elif "causal-lm-sanity" in cfg_data.task:
        loss_fn = MostlyCausalLoss()
    elif "causal-lm" in cfg_data.task:
        loss_fn = CausalLoss()
    elif "masked-lm" in cfg_data.task:
        loss_fn = MLMLoss(vocab_size=cfg_data.vocab_size)
    else:
        raise ValueError(f"No loss function registered for task {cfg_data.task}.")

    if 'smooth' in cfg_data.keys():
        if cfg_data.smooth:
            # loss_fn = CustomMSELoss(smooth=cfg_data.smooth)
            loss_fn = NewCrossEntropy(alpha=cfg_data.smooth)
            print('-----------------using label noise defense------------------')

    if 'mix' in cfg_data.keys():
        if cfg_data.mix:
            loss_fn = MixupCrossEntropy()

    # loss_fn = torch.jit.script(loss_fn)
    return model, loss_fn


def _construct_text_model(cfg_model, cfg_data, pretrained=True, **kwargs):
    if cfg_model == "transformer3f":
        # This is the transformer from "A field guide to federated learning"
        """
        we train a modified 3-layer Transformer model [250],
        where the dimension of the token embeddings is 96, and the hidden dimension of the feed-forward
        network (FFN) block is 1536. We use 8 heads for the multi-head attention, where each head is based
        on 12-dimensional (query, key, value) vectors. We use ReLU activation and set dropout rate to 0.1.
        """
        # For simplicity the dropout is disabled for now
        # the 12-dim query is 96/8 = 12
        model = TransformerModel(
            ntokens=cfg_data.vocab_size, ninp=96, nhead=8, nhid=1536, nlayers=3, dropout=0, positional_embedding="fixed"
        )
    elif cfg_model == "transformer3":
        # Same as above but with learnable positional embeddings
        model = TransformerModel(
            ntokens=cfg_data.vocab_size,
            ninp=96,
            nhead=8,
            nhid=1536,
            nlayers=3,
            dropout=0,
            positional_embedding="learnable",
        )
    elif cfg_model == "transformer3t":
        # Same as above but with learnable positional embeddings and tied weights
        model = TransformerModel(
            ntokens=cfg_data.vocab_size,
            ninp=96,
            nhead=8,
            nhid=1536,
            nlayers=3,
            dropout=0,
            positional_embedding="learnable",
            tie_weights=True,
        )
    elif cfg_model == "transformer1":
        # This is our initial sanity check transformer:
        model = TransformerModel(ntokens=cfg_data.vocab_size, ninp=200, nhead=1, nhid=200, nlayers=1, dropout=0)
    elif cfg_model == "transformerS":
        # A wide sanity-check transformer
        model = TransformerModel(ntokens=cfg_data.vocab_size, ninp=512, nhead=1, nhid=512, nlayers=1, dropout=0)
    elif cfg_model == "LSTM":
        # This is the LSTM from "LEARNING DIFFERENTIALLY PRIVATE RECURRENT LANGUAGE MODELS"
        """
        word s t is mapped to an embedding vector e t ∈ R 96
        by looking up the word in the model’s vocabulary. The e t is composed with the state emitted by
        the model in the previous time step s t−1 ∈ R 256 to emit a new state vector s t and an “output
        embedding” o t ∈ R 96 .
        """
        model = RNNModel("LSTM", cfg_data.vocab_size, ninp=96, nhid=96, nlayers=1, dropout=0.0, tie_weights=True)
    elif cfg_model == "linear":
        model = LinearModel(cfg_data.vocab_size, embedding_size=200)
    else:
        try:
            from transformers import (
                AutoModelForMaskedLM,
                AutoModelForPreTraining,
                AutoModelForSequenceClassification,
                AutoConfig,
            )

            if cfg_data.task == "masked-lm":
                auto_class = AutoModelForMaskedLM
            elif cfg_data.task == "classification":
                auto_class = AutoModelForSequenceClassification
            else:
                auto_class = AutoModelForPreTraining
            # Make sure to use the matching tokenizer and vocab_size!
            if cfg_model == "gpt2S":
                cfg_model = "gpt2"
                extra_args = dict(activation_function="relu", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
            elif cfg_model == "bert-sanity-check":
                cfg_model = "bert-base-uncased"
                extra_args = dict(hidden_act="relu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
            else:
                extra_args = dict()
            if pretrained:
                model = auto_class.from_pretrained(cfg_model, **extra_args)
            else:
                hf_cfg = AutoConfig.from_pretrained(cfg_model, **extra_args)
                model = auto_class.from_config(hf_cfg)
            # model.transformer.h[0].attn.scale_attn_weights = False
            if model.config.vocab_size != cfg_data.vocab_size:
                model.resize_token_embeddings(new_num_tokens=cfg_data.vocab_size)
            model = HuggingFaceContainer(model)
        except OSError as error_msg:
            raise ValueError(f"Invalid huggingface model {cfg_model} given: {error_msg}")
    return model


class HuggingFaceContainer(torch.nn.Module):
    """Wrap huggingface models for a unified interface. Ugh."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        if "inputs" in kwargs:
            kwargs["input_ids"] = kwargs.pop("inputs")
        if "input_ids" not in kwargs:
            kwargs["input_ids"] = args[0]
        if kwargs["input_ids"].dtype != torch.long:
            kwargs["inputs_embeds"] = kwargs.pop("input_ids")
        outputs = self.model(**kwargs)
        return outputs["logits"] if "logits" in outputs else outputs["prediction_logits"]


class VisionContainer(torch.nn.Module):
    """We'll use a container to catch extra attributes and allow for usage with model(**data)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, **kwargs):
        tmp = self.model(inputs)
        # if hasattr(tmp, 'logits'):
        #     return tmp.logits
        return tmp

    def mix_feature_forward(self, inputs, **kwargs):
        # only test for vggnet
        assert inputs.shape[0] == 2
        x = self.model.features(inputs)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        # mask = torch.rand_like(x[0]) > 0.2
        y = torch.zeros_like(x)
        # y[0] = x[0]*mask + x[1]*(~mask)
        # y[1] = x[1]*mask + x[0]*(~mask)
        alpha = kwargs['labels']['lambda'][0]
        y[0] = x[0]*alpha + x[1]*(1-alpha)
        y[1] = x[1]*alpha + x[0]*(1-alpha)
        return self.model.classifier(y)


def _construct_vision_model(cfg_model, cfg_data, model=None, pretrained=True, **kwargs):
    """Construct the neural net that is used."""
    if model is None:
        classes = cfg_data.classes
        model = getattr(torchvision.models, cfg_model.lower())(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, classes)
    return VisionContainer(model)


class ConvNetSmall(torch.nn.Module):
    """ConvNet without BN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv0", torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                    ("relu0", torch.nn.ReLU()),
                    ("conv1", torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                    ("relu1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, stride=2, padding=1)),
                    ("relu2", torch.nn.ReLU()),
                    ("pool0", torch.nn.MaxPool2d(3)),
                    ("conv3", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, stride=2, padding=1)),
                    ("relu3", torch.nn.ReLU()),
                    ("pool1", torch.nn.AdaptiveAvgPool2d(1)),
                    ("flatten", torch.nn.Flatten()),
                    ("linear", torch.nn.Linear(4 * width, num_classes)),
                ]
            )
        )

    def forward(self, input):
        return self.model(input)


class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv0", torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                    ("bn0", torch.nn.BatchNorm2d(1 * width)),
                    ("relu0", torch.nn.ReLU()),
                    ("conv1", torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                    ("bn1", torch.nn.BatchNorm2d(2 * width)),
                    ("relu1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
                    ("bn2", torch.nn.BatchNorm2d(2 * width)),
                    ("relu2", torch.nn.ReLU()),
                    ("conv3", torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn3", torch.nn.BatchNorm2d(4 * width)),
                    ("relu3", torch.nn.ReLU()),
                    ("conv4", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn4", torch.nn.BatchNorm2d(4 * width)),
                    ("relu4", torch.nn.ReLU()),
                    ("conv5", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn5", torch.nn.BatchNorm2d(4 * width)),
                    ("relu5", torch.nn.ReLU()),
                    ("pool0", torch.nn.MaxPool2d(3)),
                    ("conv6", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn6", torch.nn.BatchNorm2d(4 * width)),
                    ("relu6", torch.nn.ReLU()),
                    ("conv7", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn7", torch.nn.BatchNorm2d(4 * width)),
                    ("relu7", torch.nn.ReLU()),
                    ("pool1", torch.nn.MaxPool2d(3)),
                    ("flatten", torch.nn.Flatten()),
                    ("linear", torch.nn.Linear(36 * width, num_classes)),
                ]
            )
        )

    def forward(self, input):
        return self.model(input)


class LeNetZhu(torch.nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = torch.nn.Sigmoid
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = torch.nn.Sequential(torch.nn.Linear(768, num_classes))
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class _Select(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, : self.n]


class ModifiedBlock(torch.nn.Module):
    def __init__(self, old_Block):
        super().__init__()
        self.attn = old_Block.attn
        self.drop_path = old_Block.drop_path
        self.norm2 = old_Block.norm2
        self.mlp = old_Block.mlp

    def forward(self, x):
        x = self.attn(x)
        x = self.drop_path(self.mlp((self.norm2(x))))
        return x


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def replace_ReLU2Sigmoid(c):
    for name, module in c.named_modules():
        if isinstance(module, torch.nn.ReLU):
            _set_module(c, name, torch.nn.Sigmoid())

def replace_ReLU2Tanh(c):
    for name, module in c.named_modules():
        if isinstance(module, torch.nn.ReLU):
            _set_module(c, name, torch.nn.Tanh())


def replace_ReLU2ELU(c):
    for name, module in c.named_modules():
        if isinstance(module, torch.nn.ReLU):
            _set_module(c, name, torch.nn.ELU(alpha=1.0, inplace=True))


def replace_ReLU2ID(c):
    for name, module in c.named_modules():
        if isinstance(module, torch.nn.ReLU):
            _set_module(c, name, torch.nn.Identity())


def remove_dropout(c):
    for name, module in c.named_modules():
        if isinstance(module, torch.nn.Dropout):
            _set_module(c, name, torch.nn.Dropout(0.0, inplace=True))


def replace_ReLU62ELU(c):
    for name, module in c.named_modules():
        if isinstance(module, torch.nn.ReLU6) or isinstance(module, torch.nn.ReLU):
            _set_module(c, name, torch.nn.ELU(alpha=1.0, inplace=True))


def replace_ReLU2LeakyReLU(c):
    for name, module in c.named_modules():
        if isinstance(module, torch.nn.ReLU):
            _set_module(c, name, torch.nn.LeakyReLU(negative_slope=0.8, inplace=True))


def rand_set_batchnorm_para(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for nam, para in module.named_parameters():
                if 'weight' in nam:
                    with torch.no_grad():
                        para.data = torch.randn_like(para)
                if 'bias' in nam:
                    with torch.no_grad():
                        para.data = para.data + 10
