"""Implement some additional loss functions."""

import torch
from typing import Optional
import torch.nn.functional as F


class CausalLoss(torch.nn.Module):
    """Cross Entropy variant for next-token prediction in causal language modeling."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """If no labels are given, then the same sequence is re-used."""
        # Based on https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L1069
        # Shift so that tokens < n predict n
        shift_logits = outputs[:, :-1, :].contiguous()
        if labels is None:
            shift_labels = outputs[:, 1:].contiguous()
        elif labels.dtype == torch.long:
            shift_labels = labels[:, 1:].contiguous().view(-1)
        else:
            shift_labels = labels[:, 1:, :].contiguous().view(-1, labels.shape[-1])
        # Flatten the tokens
        return self.loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels)


class CustomMSELoss(torch.nn.Module):
    def __init__(self, smooth=0.0):
        super(CustomMSELoss, self).__init__()
        self.ss = smooth

    def forward(self, outputs, labels):
        num_class = outputs.shape[1]
        outputs = F.softmax(outputs, dim=1)
        labels = F.one_hot(labels, num_class).float()
        labels += self.ss*torch.randn_like(labels).abs_()
        labels = labels / labels.sum(dim=1, keepdim=True)
        # tmp1 = torch.max(labels[0])
        # tmp2 = torch.argmax(labels[0])
        return F.mse_loss(outputs, labels)


class NewCrossEntropy(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(NewCrossEntropy, self).__init__()
        self.a = torch.tensor([alpha, 1-alpha])
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        n = outputs.shape[0]
        lo = torch.tensor([0.]).cuda()
        tmp1 = F.softmax(outputs, dim=1)
        tmp2 = torch.max(tmp1, dim=1)
        for i in range(n):
            lo += self.loss(outputs[i:i+1], labels[i:i+1])*self.a[i]
        return lo


class MixupCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(MixupCrossEntropy, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        n = outputs.shape[0]
        if n == 1:
            loss = self._cal_mix_loss(outputs[0], labels)
            return loss
        elif n == 2:
            loss1 = self._cal_mix_loss(outputs[0], labels)
            loss2 = self._cal_mix_loss2(outputs[1], labels)
            return (loss1+loss2)/n
        elif n == 3:
            loss1 = self._cal_regular_loss(outputs[0:2], labels)
            loss2 = self._cal_mix_loss(outputs[2], labels)
            return (loss1+loss2)/n
        elif n == 4:
            loss1 = self._cal_regular_loss(outputs[0:2], labels)
            loss2 = self._cal_mix_loss(outputs[2], labels) + self._cal_mix_loss2(outputs[3], labels)
            return (loss1+loss2)/n
        else:
            raise ValueError(f'more than {n} images are used!!!')

    def _cal_mix_loss(self, output, labels):
        label1, label2 = labels['labels']
        lam1, lam2 = labels['lambda']
        logsm = F.log_softmax(output, dim=0)
        loss = -1*(logsm[label1.item()]*lam1 + logsm[label2.item()]*lam2)
        return loss

    def _cal_mix_loss2(self, output, labels):
        label1, label2 = labels['labels']
        lam1, lam2 = labels['lambda']
        logsm = F.log_softmax(output, dim=0)
        loss = -1*(logsm[label1.item()]*(1-lam1) + logsm[label2.item()]*(1-lam2))
        return loss

    def _cal_regular_loss(self, outputs, labels):
        labels = labels['labels']
        n = outputs.shape[0]
        lo = torch.tensor([0.]).cuda()
        for i in range(n):
            lo += self.loss(outputs[i:i+1], labels[i].unsqueeze(0).cuda())
        return lo


class MLMLoss(torch.nn.Module):
    def __init__(self, *args, vocab_size=50_000, **kwargs):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(*args, **kwargs)
        self.vocab_size = vocab_size

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Make sure to handle both soft labels and numeric targets."""
        # Flatten the tokens
        if labels.dtype == torch.long:
            labels = labels.view(-1)
        else:
            labels = labels.view(-1, self.vocab_size)
        return self.loss_fct(outputs.view(-1, self.vocab_size), labels)


class MostlyCausalLoss(torch.nn.Module):
    """Sanity check loss for last-token inconsistencies...
    Do not use this for anything resembling actual language model training."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """If no labels are given, then the same sequence is re-used."""
        # Based on https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L1069
        # Shift so that tokens < n predict n
        shift_logits = outputs[:, :, :].contiguous()
        if labels is None:
            shift_labels = outputs[:, 1:].contiguous()
        elif labels.dtype == torch.long:
            shift_labels = torch.cat([labels[:, 1:], labels[:, -1:]], dim=1).contiguous().view(-1)
        else:
            shift_labels = labels[:, 1:, :].contiguous().view(-1, labels.shape[-1])

        # Flatten the tokens
        return self.loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels)
