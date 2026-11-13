"""Frozen ordinary CLIP with correctly tokenized global/composite MST-PL contexts."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def standard_tokenize(texts):
    import clip
    return clip.tokenize(texts, truncate=False)


class FrozenMSTTextCLIP(nn.Module):
    def __init__(self, model, classes, tokenize=None, validate_real=True):
        super().__init__()
        self.clip = model.float().eval().requires_grad_(False)
        self.tokenize = tokenize or standard_tokenize
        self.classes = tuple(classes)
        self.context_width = int(model.ln_final.weight.shape[0])
        self.feature_width = int(model.text_projection.shape[1])
        if not classes or len(set(classes)) != len(classes) or self.context_width != self.feature_width:
            raise ValueError("MST-PL needs a unique nonempty vocabulary and aligned prompt/feature widths")
        if validate_real:
            observed = (self.context_width, self.feature_width, len(model.transformer.resblocks),
                        int(model.visual.conv1.weight.shape[0]), len(model.visual.transformer.resblocks),
                        int(model.visual.input_resolution), int(model.context_length))
            if observed != (768, 768, 12, 1024, 24, 224, 77):
                raise ValueError(f"Expected actual standard OpenAI L14/224; got {observed}")
        device = next(model.parameters()).device
        for length in (4, 8):
            texts = [" ".join(["X"] * length) + " " + name.replace("_", " ") + "." for name in classes]
            tokens = self.tokenize(texts).to(device)
            if tokens.shape[1] != model.context_length:
                raise ValueError("Tokenizer/model context lengths differ")
            with torch.no_grad():
                embedded = self.clip.token_embedding(tokens)
            self.register_buffer(f"prefix_{length}", embedded[:, :1].detach().clone())
            self.register_buffer(f"suffix_{length}", embedded[:, 1 + length:].detach().clone())
            self.register_buffer(f"tokens_{length}", tokens)
            self.register_buffer(f"eot_{length}", tokens.argmax(dim=-1))

    def train(self, mode=True):
        super().train(False)
        return self

    @property
    def scale(self):
        return self.clip.logit_scale.detach().float().exp()

    @torch.no_grad()
    def image_features(self, images):
        return F.normalize(self.clip.encode_image(images.float()).float(), dim=-1, eps=1e-8)

    @torch.no_grad()
    def handcrafted_features(self, texts):
        device = next(self.clip.parameters()).device
        tokens = self.tokenize(texts).to(device)
        return F.normalize(self.clip.encode_text(tokens).float(), dim=-1, eps=1e-8)

    def context_features(self, contexts):
        """[L,W] -> [C,D], or [B,L,W] -> [B,C,D]; true L4 or L8 EOT."""
        single = contexts.ndim == 2
        if single:
            contexts = contexts.unsqueeze(0)
        if contexts.ndim != 3 or contexts.shape[1] not in (4, 8) or contexts.shape[2] != self.context_width:
            raise ValueError("Expected true4(global) or8(composite) context tokens")
        batch, length, _ = contexts.shape
        classes = len(self.classes)
        prefix = getattr(self, f"prefix_{length}").unsqueeze(0).expand(batch, -1, -1, -1)
        suffix = getattr(self, f"suffix_{length}").unsqueeze(0).expand(batch, -1, -1, -1)
        ctx = contexts[:, None].expand(-1, classes, -1, -1)
        x = torch.cat((prefix, ctx, suffix), dim=2).reshape(batch * classes, self.clip.context_length, -1)
        x = x + self.clip.positional_embedding.float()
        x = self.clip.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.clip.ln_final(x).float()
        eot = getattr(self, f"eot_{length}").repeat(batch)
        features = x[torch.arange(batch * classes, device=x.device), eot] @ self.clip.text_projection.float()
        features = F.normalize(features.float(), dim=-1, eps=1e-8).reshape(batch, classes, -1)
        return features[0] if single else features

    def logits(self, image_features, global_prompt, domain_prompt=None):
        contexts = global_prompt if domain_prompt is None else torch.cat((global_prompt, domain_prompt), dim=0)
        text = self.context_features(contexts)
        return self.scale * F.normalize(image_features.float(), dim=-1, eps=1e-8) @ text.t()

    def dynamic_logits(self, features, global_prompt, domain_prompts, domain_classifier):
        features = F.normalize(features.float(), dim=-1, eps=1e-8)
        weights = F.softmax(domain_classifier(features), dim=-1)
        generated = torch.einsum("bk,kld->bld", weights, domain_prompts)
        global_expanded = global_prompt.unsqueeze(0).expand(len(features), -1, -1)
        text = self.context_features(torch.cat((global_expanded, generated), dim=1))
        return self.scale * torch.einsum("bd,bcd->bc", features, text), weights
