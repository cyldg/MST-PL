"""Source-only, label-free reconstruction of the manuscript's unsupervised DPL.

The caller supplies two frozen-CLIP features for each *source training* image.
There is deliberately no class ground-truth or target-data argument. Class names,
domain identities and handcrafted text features are public task information.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


LABEL = "manuscript-unsupervised-MST-PL-DPL-reconstruction-20261103"


def freeze(module):
    module.eval().requires_grad_(False)
    for parameter in module.parameters():
        parameter.grad = None


def finite_step(loss, parameters, optimizer):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite DPL loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Missing or nonfinite active-stage gradients")
    optimizer.step()


def equal_client_mean(values):
    if not values or any(value.shape != values[0].shape for value in values):
        raise ValueError("Equal aggregation requires matching nonempty client uploads")
    return torch.stack([value.detach() for value in values]).mean(0).clone()


class StyleMLP(nn.Module):
    def __init__(self, width=768, hidden=384):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, features):
        return self.fc2(F.relu(self.fc1(features)))


def unsupervised_style_objective(output, original, pseudo_labels, source_text,
                                 destination_text, teacher_probas, scale, lam=.5):
    """Pseudo-label direction + KL(P_transformed || P_original), no hard class CE."""
    direction = F.normalize(destination_text[pseudo_labels] - source_text[pseudo_labels], dim=-1, eps=1e-8)
    displacement = F.normalize(output - original, dim=-1, eps=1e-8)
    alignment = (1 - (displacement * direction).sum(-1)).mean()
    log_aug = F.log_softmax(scale * F.normalize(output, dim=-1, eps=1e-8) @ source_text.t(), dim=-1)
    log_teacher = teacher_probas.detach().clamp_min(torch.finfo(output.dtype).tiny).log()
    consistency = (log_aug.exp() * (log_aug - log_teacher)).sum(-1).mean()
    return lam * alignment + (1 - lam) * consistency, alignment, consistency


def paper_cross_view(logits_weak, logits_strong, teacher_weak, teacher_strong):
    """The literal printed CE(p,q)=-sum p log q, including its first-input gradient."""
    p_weak = F.softmax(logits_weak, dim=-1)
    log_strong = F.log_softmax(logits_strong, dim=-1)
    tiny = torch.finfo(logits_weak.dtype).tiny
    return (-(p_weak * teacher_strong.detach().clamp_min(tiny).log()).sum(-1)
            - (teacher_weak.detach() * log_strong).sum(-1)).mean()


def domain_contrast(domain_prompt, global_prompt, handcrafted):
    pd = F.normalize(domain_prompt.mean(0), dim=-1, eps=1e-8)
    pg = F.normalize(global_prompt.detach().mean(0), dim=-1, eps=1e-8)
    ti = F.normalize(handcrafted.detach(), dim=-1, eps=1e-8)
    logits = torch.stack((pd @ ti, pd @ pg)).unsqueeze(0)
    return F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=logits.device))


@dataclass(frozen=True)
class DPLConfig:
    rounds: int = 20
    global_epochs: int = 5
    domain_epochs: int = 5
    mst_epochs: int = 50  # Recovered source convention; not specified in manuscript.
    batch_size: int = 16
    prompt_lr: float = .005
    classifier_lr: float = .01
    mst_lr: float = .001
    mst_weight_decay: float = .05
    mst_hidden: int = 384
    mst_lambda: float = .5
    alpha_uce: float = .5  # Recovered source; absent from manuscript.
    beta_cv: float = .5
    initial_threshold: float = .90
    purification_threshold: float = .88
    expansion_threshold: float = .92
    variant: str = "full"
    max_batches: int = 0  # Nonzero is an explicitly truncated diagnostic, not formal.

    def __post_init__(self):
        if self.variant not in ("full", "initial_fixed", "expansion_only"):
            raise ValueError("Unknown pseudo-label policy")
        if min(self.rounds, self.global_epochs, self.domain_epochs, self.mst_epochs,
               self.batch_size, self.mst_hidden) < 1 or self.max_batches < 0:
            raise ValueError("Training budgets must be positive")
        if not (0 <= self.mst_lambda <= 1 and 0 <= self.purification_threshold <= 1
                and 0 <= self.initial_threshold <= 1 and 0 <= self.expansion_threshold <= 1):
            raise ValueError("Invalid coefficient or threshold")
        if min(self.prompt_lr, self.classifier_lr, self.mst_lr) <= 0:
            raise ValueError("Learning rates must be positive")


@dataclass(frozen=True)
class ClientFeatures:
    sample_ids: tuple[str, ...]
    weak: torch.Tensor
    strong: torch.Tensor


@dataclass
class PseudoState:
    selected: torch.Tensor
    labels: torch.Tensor

    def clone(self):
        return PseudoState(self.selected.clone(), self.labels.clone())


class FederatedDPL:
    """Frozen-feature MST, alternating prompts and dynamic source pseudo labels.

    ``bridge`` is FrozenMSTTextCLIP-compatible: widths, scale, context_features,
    logits and dynamic_logits. ``observer(event, client, round, stage, **payload)``
    receives CPU tensor snapshots, never GT. No target loader is accepted here.
    """
    def __init__(self, bridge, clients: Mapping, domain_text: Mapping,
                 handcrafted: Mapping, config=DPLConfig(), seed=0, device="cpu",
                 observer: Callable | None = None):
        self.config, self.seed, self.device = config, int(seed), torch.device(device)
        self.bridge = bridge.to(self.device)
        freeze(self.bridge)
        self.observer = observer
        self.sources = tuple(clients)
        if len(self.sources) < 2 or len(set(self.sources)) != len(self.sources):
            raise ValueError("At least two unique source clients are required")
        if set(domain_text) != set(self.sources) or set(handcrafted) != set(self.sources):
            raise ValueError("Handcrafted text must cover exactly the source domains")
        if bridge.context_width != bridge.feature_width:
            raise ValueError("Domain contrast requires matching CLIP context/feature widths")
        self.clients, self.domain_text, self.handcrafted = {}, {}, {}
        self.teacher_weak, self.teacher_strong, self.manifest = {}, {}, {}
        class_count = None
        for source in self.sources:
            value = clients[source]
            if isinstance(value, Mapping):
                if set(value) != {"sample_ids", "weak", "strong"}:
                    raise ValueError("Clients accept only sample_ids, weak and strong; no label/target fields")
                value = ClientFeatures(tuple(value["sample_ids"]), value["weak"], value["strong"])
            ids = tuple(value.sample_ids)
            if not ids or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
                raise ValueError("Stable nonempty unique sample IDs required in each source")
            weak, strong = (self._feature_tensor(x, len(ids)) for x in (value.weak, value.strong))
            self.clients[source] = ClientFeatures(ids, weak, strong)
            text = domain_text[source].detach().float().to(self.device)
            if (text.ndim != 2 or text.shape[1] != bridge.feature_width or text.shape[0] < 2
                    or not bool(torch.isfinite(text).all()) or bool((text.norm(dim=-1) == 0).any())):
                raise ValueError("Expected domain-specific [classes,feature_width] text features")
            if class_count is not None and class_count != len(text):
                raise ValueError("All source clients must share the same ordered class vocabulary")
            class_count = len(text)
            self.domain_text[source] = F.normalize(text, dim=-1, eps=1e-8)
            hand = handcrafted[source].detach().float().to(self.device).reshape(-1).clone()
            if hand.shape != (bridge.feature_width,) or not bool(torch.isfinite(hand).all()) or hand.norm() == 0:
                raise ValueError("Expected one frozen style-only handcrafted domain text vector")
            self.handcrafted[source] = hand
            digest = hashlib.sha256()
            digest.update(json.dumps(ids, ensure_ascii=False).encode())
            digest.update(weak.numpy().tobytes()); digest.update(strong.numpy().tobytes())
            digest.update(self.domain_text[source].cpu().numpy().tobytes())
            digest.update(hand.cpu().numpy().tobytes())
            self.manifest[source] = digest.hexdigest()
            self.teacher_weak[source] = self._teacher_probabilities(weak, source)
            self.teacher_strong[source] = self._teacher_probabilities(strong, source)
        self.class_count = class_count
        if hasattr(bridge, "classes") and len(bridge.classes) != class_count:
            raise ValueError("Bridge and handcrafted class vocabularies differ")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed("initialization"))
            self.global_prompt = torch.empty(4, bridge.context_width).normal_(std=.02).to(self.device)
            self.domain_prompts = [nn.Parameter(torch.empty(4, bridge.context_width).normal_(std=.02).to(self.device),
                                               requires_grad=False) for _ in self.sources]
            self.domain_classifier = nn.Linear(bridge.feature_width, len(self.sources), bias=False).to(self.device)
        freeze(self.domain_classifier)
        self.states, self.initial_states, self.transforms, self.augmented = {}, {}, {}, {}
        self.initialized = False
        self.complete_rounds = 0
        self.updates = dict.fromkeys(("MST", "global_prompt", "domain_classifier", "domain_prompt"), 0)
        self.samples_seen = dict.fromkeys(self.updates, 0)
        self.mst_events = []

    def _feature_tensor(self, value, n):
        value = value.detach().float().cpu().contiguous()
        if value.shape != (n, self.bridge.feature_width) or not bool(torch.isfinite(value).all()):
            raise ValueError("Each source needs aligned finite weak/strong [N,feature_width] features")
        if bool((value.norm(dim=-1) == 0).any()):
            raise ValueError("Zero image features are invalid")
        return F.normalize(value, dim=-1, eps=1e-8).contiguous()

    def _seed(self, stage, round=0, client=0, epoch=0, destination=0):
        encoded = f"{self.seed}:{stage}:{round}:{client}:{epoch}:{destination}".encode()
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)

    def _batches(self, n, stage, client, epoch, eligible=None, destination=0):
        # Permute immutable full-cohort positions BEFORE filtering. Thus common
        # samples retain their relative order when policies choose different masks.
        order = torch.randperm(n, generator=torch.Generator().manual_seed(
            self._seed(stage, self.complete_rounds + 1, client, epoch, destination)))
        if eligible is not None:
            order = order[eligible[order]]
        for batch_index, positions in enumerate(order.split(self.config.batch_size)):
            if self.config.max_batches and batch_index >= self.config.max_batches:
                break
            if len(positions):
                yield positions

    @torch.no_grad()
    def _teacher_probabilities(self, features, source):
        result = []
        for x in features.split(self.config.batch_size):
            logits = self.bridge.scale * x.to(self.device) @ self.domain_text[source].t()
            result.append(logits.softmax(-1).cpu())
        return torch.cat(result)

    def _emit(self, event, source, round, stage, **payload):
        if self.observer is not None:
            snapshot = {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                        for key, value in payload.items()}
            self.observer(event, source, round, stage, **snapshot)

    def _emit_state(self, source, round, stage, probabilities, strong_probabilities=None):
        confidence, candidate = probabilities.max(-1)
        state = self.states[source]
        self._emit("state", source, round, stage, selected=state.selected, labels=state.labels,
                   candidate=candidate, confidence=confidence,
                   view0_label=candidate, view1_label=(strong_probabilities.argmax(-1)
                                                      if strong_probabilities is not None else None))

    def initialize_pseudo_labels(self):
        if self.initialized:
            raise RuntimeError("PLIF may run only once; load a shared initialization instead")
        for source in self.sources:
            confidence, labels = self.teacher_weak[source].max(-1)
            selected = confidence > self.config.initial_threshold
            labels = torch.where(selected, labels, torch.full_like(labels, -1))
            self.states[source] = PseudoState(selected.clone(), labels.clone())
            self.initial_states[source] = self.states[source].clone()
        self.initialized = True
        self.emit_initial_observations()
        return {source: int(state.selected.sum()) for source, state in self.states.items()}

    def emit_initial_observations(self):
        """Replay r0 after loading a shared PLIF+MST checkpoint into a new trace."""
        if not self.initialized or self.complete_rounds != 0:
            raise RuntimeError("Initial observations require an initialized pre-round state")
        for source in self.sources:
            self._emit_state(source, 0, "plif", self.teacher_weak[source], self.teacher_strong[source])
        self.emit_probes(0)

    def prepare_style(self):
        if not self.initialized:
            raise RuntimeError("Run PLIF before unsupervised MST")
        if self.transforms or self.complete_rounds:
            raise RuntimeError("MST is fitted once and remains fixed throughout DPL")
        empty = [source for source in self.sources if not bool(self.initial_states[source].selected.any())]
        if empty:
            raise ValueError(f"Empty PLIF source(s) {empty}; refusing threshold fallback or random MST")
        cfg = self.config
        for i, source in enumerate(self.sources):
            features = self.clients[source].weak
            state = self.initial_states[source]
            usage = torch.zeros(len(features), dtype=torch.long)
            for j, destination in enumerate(self.sources):
                if i == j:
                    continue
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self._seed("MST-initialization", client=i, destination=j))
                    net = StyleMLP(self.bridge.feature_width, cfg.mst_hidden).to(self.device)
                optimizer = torch.optim.Adam(net.parameters(), lr=cfg.mst_lr, weight_decay=cfg.mst_weight_decay)
                last, steps = None, 0
                for epoch in range(cfg.mst_epochs):
                    for positions in self._batches(len(features), "MST", i, epoch, state.selected, j):
                        x = features[positions].to(self.device)
                        y = state.labels[positions].to(self.device)
                        p = self.teacher_weak[source][positions].to(self.device)
                        loss, alignment, consistency = unsupervised_style_objective(
                            net(x), x, y, self.domain_text[source], self.domain_text[destination],
                            p, self.bridge.scale, cfg.mst_lambda)
                        finite_step(loss, list(net.parameters()), optimizer)
                        usage[positions] += 1
                        self.updates["MST"] += 1; self.samples_seen["MST"] += len(positions)
                        steps += 1
                        last = {"total": float(loss.detach()), "alignment": float(alignment.detach()),
                                "KL_aug_to_original": float(consistency.detach())}
                if not steps:
                    raise RuntimeError("MST received no selected examples")
                freeze(net)
                self.transforms[(i, j)] = net
                self.mst_events.append({"source": source, "destination_source_domain": destination,
                                        "epochs": cfg.mst_epochs, "updates": steps, "last_losses": last})
            self._emit("usage", source, 0, "mst_alignment", usage_counts=usage, pseudo_labels_used=state.labels)
        self._build_augmentation_banks()
        return list(self.mst_events)

    @torch.no_grad()
    def _build_augmentation_banks(self):
        expected = {(i, j) for i in range(len(self.sources)) for j in range(len(self.sources)) if i != j}
        if set(self.transforms) != expected:
            raise ValueError("MST transforms must cover all ordered pairs of source domains only")
        self.augmented = {}
        for i, source in enumerate(self.sources):
            original = self.clients[source].weak
            features, rows, domains = [original], [torch.arange(len(original))], [torch.full((len(original),), i)]
            for j in range(len(self.sources)):
                if i == j:
                    continue
                net = self.transforms[(i, j)]
                generated = [F.normalize(net(x.to(self.device)), dim=-1, eps=1e-8).cpu()
                             for x in original.split(self.config.batch_size)]
                features.append(torch.cat(generated)); rows.append(torch.arange(len(original)))
                domains.append(torch.full((len(original),), j))
            # Labels are intentionally absent. Every training batch indexes the
            # current label table by immutable source row, including MST copies.
            bank = {"features": torch.cat(features), "row_ids": torch.cat(rows), "domains": torch.cat(domains)}
            if not bool(torch.isfinite(bank["features"]).all()):
                raise FloatingPointError("Nonfinite fixed MST augmentation bank")
            self.augmented[source] = bank

    def _prompt_optimizer(self, parameters):
        return torch.optim.SGD(parameters, lr=self.config.prompt_lr, momentum=0., weight_decay=0.)

    def _global_stage(self, round):
        cfg = self.config
        global_uploads, classifier_uploads, metrics = [], [], {}
        for i, source in enumerate(self.sources):
            pg = nn.Parameter(self.global_prompt.detach().clone())
            classifier = nn.Linear(self.bridge.feature_width, len(self.sources), bias=False).to(self.device)
            classifier.load_state_dict(self.domain_classifier.state_dict())
            pg_optimizer = self._prompt_optimizer([pg])
            dc_optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.classifier_lr, momentum=0., weight_decay=0.)
            bank, state = self.augmented[source], self.states[source]
            eligible = state.selected[bank["row_ids"]]
            usage = torch.zeros(len(state.labels), dtype=torch.long)
            pg_steps = dc_steps = 0
            for epoch in range(cfg.global_epochs):
                for positions in self._batches(len(eligible), "global", i, epoch, eligible):
                    rows = bank["row_ids"][positions]
                    labels = state.labels[rows].to(self.device)
                    x = bank["features"][positions].to(self.device)
                    loss = F.cross_entropy(self.bridge.logits(x, pg), labels)
                    finite_step(loss, [pg], pg_optimizer)
                    usage.scatter_add_(0, rows, torch.ones_like(rows))
                    pg_steps += 1; self.updates["global_prompt"] += 1
                    self.samples_seen["global_prompt"] += len(rows)
                for positions in self._batches(len(eligible), "classifier", i, epoch, eligible):
                    x = bank["features"][positions].to(self.device)
                    domains = bank["domains"][positions].to(self.device)
                    loss = F.cross_entropy(classifier(x), domains)
                    finite_step(loss, list(classifier.parameters()), dc_optimizer)
                    dc_steps += 1; self.updates["domain_classifier"] += 1
                    self.samples_seen["domain_classifier"] += len(domains)
            global_uploads.append(pg.detach().clone())
            classifier_uploads.append(classifier.weight.detach().clone())
            metrics[source] = {"global_steps": pg_steps, "classifier_steps": dc_steps}
            self._emit("usage", source, round, "global_class_ce", usage_counts=usage, pseudo_labels_used=state.labels)
        self.global_prompt = equal_client_mean(global_uploads)
        with torch.no_grad():
            self.domain_classifier.weight.copy_(equal_client_mean(classifier_uploads))
        freeze(self.domain_classifier)
        return metrics

    @torch.no_grad()
    def cohort_probabilities(self, source, composite=False, strong=False):
        i = self.sources.index(source)
        features = self.clients[source].strong if strong else self.clients[source].weak
        pd = self.domain_prompts[i] if composite else None
        text = self.bridge.context_features(self.global_prompt if pd is None
                                            else torch.cat((self.global_prompt, pd), dim=0))
        results = [F.softmax(self.bridge.scale * x.to(self.device) @ text.t(), dim=-1).cpu()
                   for x in features.split(self.config.batch_size)]
        return torch.cat(results)

    def purify(self, source, probabilities):
        """Only the preceding active cohort can be retained/relabelled by PLP."""
        if self.config.variant != "full":
            return
        confidence, candidate = probabilities.detach().cpu().max(-1)
        state = self.states[source]
        keep = state.selected & (confidence >= self.config.purification_threshold)
        state.selected = keep
        state.labels = torch.where(keep, candidate, torch.full_like(candidate, -1))

    def expand(self, source, probabilities, strong_probabilities):
        if self.config.variant == "initial_fixed":
            return
        confidence, candidate = probabilities.detach().cpu().max(-1)
        strong_candidate = strong_probabilities.detach().cpu().argmax(-1)
        state = self.states[source]
        add = ~state.selected & (confidence >= self.config.expansion_threshold) & (candidate == strong_candidate)
        state.labels[add] = candidate[add]
        state.selected[add] = True

    def _domain_stage(self, round, metrics):
        cfg = self.config
        for i, source in enumerate(self.sources):
            pd, client, state = self.domain_prompts[i], self.clients[source], self.states[source]
            pd.requires_grad_(True)
            optimizer = self._prompt_optimizer([pd])
            usage = torch.zeros(len(client.sample_ids), dtype=torch.long)
            steps, last = 0, None
            for epoch in range(cfg.domain_epochs):
                for positions in self._batches(len(client.sample_ids), "domain", i, epoch):
                    # One text pass serves aligned weak/strong image features.
                    text = self.bridge.context_features(torch.cat((self.global_prompt, pd), dim=0))
                    weak = client.weak[positions].to(self.device)
                    strong = client.strong[positions].to(self.device)
                    lw, ls = self.bridge.scale * weak @ text.t(), self.bridge.scale * strong @ text.t()
                    active = state.selected[positions]
                    if bool(active.any()):
                        labels = state.labels[positions][active].to(self.device)
                        active_device = active.to(self.device)
                        # The printed objective averages UCE over the trusted set,
                        # whereas CV averages over the entire source cohort. With
                        # full-cohort batches, correct for active-set sampling so
                        # the effective alpha does not shrink with coverage.
                        scale = len(client.sample_ids) / (int(state.selected.sum()) * len(positions))
                        uce = scale * (F.cross_entropy(lw[active_device], labels, reduction="sum")
                                       + F.cross_entropy(ls[active_device], labels, reduction="sum"))
                        usage[positions[active]] += 2
                    else:
                        uce = lw.sum() * 0.
                    cv = paper_cross_view(lw, ls, self.teacher_weak[source][positions].to(self.device),
                                          self.teacher_strong[source][positions].to(self.device))
                    contrast = domain_contrast(pd, self.global_prompt, self.handcrafted[source])
                    loss = cfg.alpha_uce * uce + contrast + cfg.beta_cv * cv
                    finite_step(loss, [pd], optimizer)
                    steps += 1; self.updates["domain_prompt"] += 1
                    self.samples_seen["domain_prompt"] += len(positions)
                    last = {"UCE": float(uce.detach()), "contrast": float(contrast.detach()),
                            "CV_paper_literal": float(cv.detach()), "total": float(loss.detach())}
            pd.requires_grad_(False); pd.grad = None
            metrics[source].update(domain_steps=steps, last_domain_losses=last)
            self._emit("usage", source, round, "domain_uce", usage_counts=usage, pseudo_labels_used=state.labels)

    @torch.no_grad()
    def emit_probes(self, round=None):
        round = self.complete_rounds if round is None else round
        for source in self.sources:
            pg = self.cohort_probabilities(source)
            pc = self.cohort_probabilities(source, composite=True)
            self._emit("probe", source, round, "round_end", global_pred=pg.argmax(-1),
                       composite_pred=pc.argmax(-1), global_confidence=pg.max(-1).values,
                       composite_confidence=pc.max(-1).values)

    def run_round(self):
        if not self.initialized or len(self.transforms) != len(self.sources) * (len(self.sources) - 1):
            raise RuntimeError("PLIF and the complete source-only unsupervised MST stage are required")
        if self.complete_rounds >= self.config.rounds:
            raise RuntimeError("Fixed final-round budget already reached")
        freeze(self.bridge)
        freeze(self.domain_classifier)
        round = self.complete_rounds + 1
        metrics = self._global_stage(round)
        for source in self.sources:
            p = self.cohort_probabilities(source)
            self._emit_state(source, round, "global_before_plp", p)
            self.purify(source, p)
            self._emit_state(source, round, "plp", p)
        self._domain_stage(round, metrics)
        for source in self.sources:
            p = self.cohort_probabilities(source, composite=True)
            ps = self.cohort_probabilities(source, composite=True, strong=True)
            self.expand(source, p, ps)
            self._emit_state(source, round, "ple", p, ps)
        self.complete_rounds = round
        self.emit_probes()
        return {"complete_round": round, "variant": self.config.variant, "clients": metrics,
                "selected": {source: int(self.states[source].selected.sum()) for source in self.sources},
                "aggregation_weights": [1 / len(self.sources)] * len(self.sources),
                "phase_order": ["global_and_classifier", "equal_aggregation_broadcast", "PLP", "local_domain", "PLE"]}

    @torch.no_grad()
    def predict(self, features):
        """Final inductive inference; callers enforce their separate target-evaluation gate."""
        if self.complete_rounds != self.config.rounds:
            raise RuntimeError("Inductive inference is available only at the fixed final round")
        features = F.normalize(features.to(self.device).float(), dim=-1, eps=1e-8)
        return self.bridge.dynamic_logits(features, self.global_prompt,
                                         torch.stack([p.detach() for p in self.domain_prompts]), self.domain_classifier)

    def checkpoint(self):
        return {"implementation_label": LABEL, "config": asdict(self.config), "seed": self.seed,
                "sources": list(self.sources), "feature_manifest": dict(self.manifest),
                "sample_ids": {s: list(self.clients[s].sample_ids) for s in self.sources},
                "complete_rounds": self.complete_rounds, "initialized": self.initialized,
                "global_prompt": self.global_prompt.detach().cpu().clone(),
                "domain_prompts": torch.stack([p.detach().cpu() for p in self.domain_prompts]),
                "domain_classifier": {k: v.detach().cpu().clone() for k, v in self.domain_classifier.state_dict().items()},
                "states": {s: {"selected": v.selected.clone(), "labels": v.labels.clone()} for s, v in self.states.items()},
                "initial_states": {s: {"selected": v.selected.clone(), "labels": v.labels.clone()} for s, v in self.initial_states.items()},
                "MST_transforms": {f"{i}->{j}": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                                   for (i, j), m in self.transforms.items()},
                "updates": dict(self.updates), "samples_seen": dict(self.samples_seen), "mst_events": list(self.mst_events)}

    def load_checkpoint(self, value):
        if (value["implementation_label"] != LABEL or value["sources"] != list(self.sources)
                or value["feature_manifest"] != self.manifest or value["seed"] != self.seed
                or value["sample_ids"] != {s: list(self.clients[s].sample_ids) for s in self.sources}):
            raise ValueError("Checkpoint source/features/seed identity mismatch")
        requested, saved = asdict(self.config), dict(value["config"])
        # Three policies may branch only from the same pre-round PLIF+MST state.
        if requested["variant"] != saved["variant"] and value["complete_rounds"] != 0:
            raise ValueError("Policy may change only before federated round one")
        requested.pop("variant"); saved.pop("variant")
        if requested != saved:
            raise ValueError("Checkpoint training settings differ from the active configuration")
        if value["global_prompt"].shape != self.global_prompt.shape or value["domain_prompts"].shape != (len(self.sources), *self.global_prompt.shape):
            raise ValueError("Prompt checkpoint dimensions differ")
        self.global_prompt = value["global_prompt"].to(self.device).detach().clone()
        for p, loaded in zip(self.domain_prompts, value["domain_prompts"]):
            with torch.no_grad():
                p.copy_(loaded.to(self.device))
            p.requires_grad_(False)
        self.domain_classifier.load_state_dict(value["domain_classifier"], strict=True)
        freeze(self.domain_classifier)
        for field in ("states", "initial_states"):
            restored = {}
            if value["initialized"] and set(value[field]) != set(self.sources):
                raise ValueError("Initialized checkpoint must contain every source pseudo-state")
            for source, data in value[field].items():
                selected, labels = data["selected"].cpu().bool().clone(), data["labels"].cpu().long().clone()
                if selected.shape != (len(self.clients[source].sample_ids),) or labels.shape != selected.shape:
                    raise ValueError("Pseudo-state shape mismatch")
                if bool((labels[~selected] != -1).any()) or bool(((labels[selected] < 0) | (labels[selected] >= self.class_count)).any()):
                    raise ValueError("Invalid checkpoint pseudo labels")
                restored[source] = PseudoState(selected, labels)
            setattr(self, field, restored)
        self.transforms = {}
        for key, state in value["MST_transforms"].items():
            i, j = map(int, key.split("->"))
            if i == j or not (0 <= i < len(self.sources) and 0 <= j < len(self.sources)):
                raise ValueError("Checkpoint contains a non-source MST transform")
            net = StyleMLP(self.bridge.feature_width, self.config.mst_hidden).to(self.device)
            net.load_state_dict(state, strict=True); freeze(net)
            self.transforms[(i, j)] = net
        self.complete_rounds, self.initialized = int(value["complete_rounds"]), bool(value["initialized"])
        if not 0 <= self.complete_rounds <= self.config.rounds:
            raise ValueError("Checkpoint round lies outside the fixed experiment budget")
        self.updates, self.samples_seen = dict(value["updates"]), dict(value["samples_seen"])
        self.mst_events = list(value["mst_events"])
        if self.transforms:
            self._build_augmentation_banks()
        else:
            self.augmented = {}
        return self
