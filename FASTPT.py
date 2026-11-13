import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip # Make sure the clip library is installed and accessible
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def c_loss(model,handcrafted_domain_prompt,device):
    P_i_D = model.prompt_learner.domain_ctx # Shape: [L_domain, Dim]
    P_G = model.prompt_learner.global_ctx   # Shape: [L_global, Dim] (current aggregated version)

    P_i_D_vec = P_i_D.mean(dim=0)
    P_G_vec = P_G.mean(dim=0)

    with torch.no_grad():
        tokenized_ti = clip.tokenize([handcrafted_domain_prompt]).to(device)
        ti_features = model.clip_model.encode_text(tokenized_ti) # Full encoding
        ti_vec = ti_features[0] # Shape: [FeatureDim]


    P_i_D_vec_norm = P_i_D_vec / (P_i_D_vec.norm(dim=-1, keepdim=True) + 1e-6)
    P_G_vec_norm = P_G_vec / (P_G_vec.norm(dim=-1, keepdim=True) + 1e-6)
    ti_vec_norm = ti_vec / (ti_vec.norm(dim=-1, keepdim=True) + 1e-6)


    sim_pd_ti = P_i_D_vec_norm @ ti_vec_norm      # Positive sim (P_i^D vs t_i)
    sim_pd_pg = P_i_D_vec_norm @ P_G_vec_norm      # Negative sim (P_i^D vs P^G)


    contrastive_logits = torch.stack([sim_pd_ti, sim_pd_pg]) / model.contrastive_temperature
    contrastive_target = torch.tensor([0], dtype=torch.long, device=device) # Target positive
    loss_contrastive = F.cross_entropy(contrastive_logits.unsqueeze(0), contrastive_target)
    return loss_contrastive

class TextEncoder(nn.Module):
    """ Encodes text prompts using CLIP's transformer. """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        eot_indices = tokenized_prompts.argmax(dim=-1)

        x = x[torch.arange(x.shape[0], device=x.device), eot_indices] @ self.text_projection

        return x

class FastPtPromptLearner(nn.Module):

    def __init__(self, n_ctx_global, n_ctx_domain, classnames, clip_model,device):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0] # Dimension of context vectors
        self.ctx_dim = ctx_dim
        self.dtype = dtype
        #初始化
        print(f"Initializing FaST-PT PromptLearner with {n_ctx_global} global and {n_ctx_domain} domain contexts.")

        global_ctx_vectors = torch.empty(n_ctx_global, ctx_dim, dtype=dtype)
        nn.init.normal_(global_ctx_vectors, std=0.02)
        self.global_ctx = nn.Parameter(global_ctx_vectors) # P^G, Shape: [L_global, Dim]
        print(f"Initial global_ctx isnan: {torch.isnan(self.global_ctx).any()}")

        domain_ctx_vectors = torch.empty(n_ctx_domain, ctx_dim, dtype=dtype)
        nn.init.normal_(domain_ctx_vectors, std=0.02)
        self.domain_ctx = nn.Parameter(domain_ctx_vectors) # P_i^D, Shape: [L_domain, Dim]

        classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = n_cls
        self.n_ctx_global = n_ctx_global
        self.n_ctx_domain = n_ctx_domain
        n_ctx_total = n_ctx_global + n_ctx_domain # Total length for composite prompts

        prompt_prefix = " ".join(["X"] * n_ctx_total) # Placeholder prefix for full prompt length
        prompts_text = [prompt_prefix + " " + name + "." for name in classnames] # e.g., "X X ... X cat."

        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_text])

        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts.to(device)).type(dtype)

        # Store prefix (SOS token embedding)
        self.register_buffer("token_prefix", embedding[:, :1, :]) 
        # Store suffix (Classname + EOS token embeddings)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx_total:, :])

        print(f"Total prompt length (including class): {embedding.shape[1]}")
        print(f"Token prefix shape: {self.token_prefix.shape}")
        print(f"Token suffix shape: {self.token_suffix.shape}")


    def forward(self, mode="composite", P_N=None):
        
        prefix = self.token_prefix # Shape: [N_cls, 1, Dim]
        suffix = self.token_suffix # Shape: [N_cls, N_suffix, Dim]

        current_device = prefix.device
        global_ctx = self.global_ctx.to(current_device) # Shape: [L_global, Dim]
        domain_ctx = self.domain_ctx.to(current_device) # Shape: [L_domain, Dim]
        # domain_NONE = " ".join(["X"] * self.domain_ctx)

        global_ctx_expanded = global_ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # [N_cls, L_global, Dim]
        domain_ctx_expanded = domain_ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # [N_cls, L_domain, Dim]
        # domain_NONE_expanded = domain_NONE.unsqueeze(0).expand(self.n_cls, -1, -1) # [N_cls, L_domain, Dim]

        if mode == "composite":
            prompts = torch.cat([
                prefix,
                global_ctx_expanded, # P^G
                domain_ctx_expanded, # P_i^D
                suffix
            ], dim=1)
            current_tokenized = self.tokenized_prompts.to(current_device)

        elif mode == "global_only":
            domain_ctx_expanded_dummy = torch.zeros_like(domain_ctx_expanded)
            prompts = torch.cat([
                prefix[:, :1, :], # Reuse SOS
                global_ctx_expanded, # P^G
                domain_ctx_expanded_dummy,
                suffix # <<< This alignment is likely incorrect for TextEncoder
            ], dim=1)
            current_tokenized = self.tokenized_prompts.to(current_device) # Placeholder
            # print("Warning: 'global_only' mode prompt length might mismatch tokenization.")


        elif mode == "inference":
            # Concatenate: [SOS, P^G, P^N, Classname, EOS]
            assert P_N is not None, "P_N must be provided for inference mode"
            P_N = P_N.to(current_device)
            if P_N.dim() == 2:
                P_N = P_N.unsqueeze(0) # Shape: [1, L_domain, Dim]
            P_N_expanded = P_N.expand(self.n_cls, -1, -1) # Shape: [N_cls, L_domain, Dim]

            prompts = torch.cat([
                prefix,
                global_ctx_expanded, # P^G
                P_N_expanded,        # P^N (Dynamic)
                suffix
            ], dim=1)
            current_tokenized = self.tokenized_prompts.to(current_device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return prompts, current_tokenized


class FaSTPT_Client(nn.Module):

    def __init__(self, classnames, clip_model,
                 n_ctx_global=8, n_ctx_domain=8, domain_number=6,
                 client_domain_id=0, contrastive_lambda=0.1, contrastive_temp=0.07,device=None):

        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

        self.prompt_learner = FastPtPromptLearner(n_ctx_global, n_ctx_domain, classnames, clip_model,device)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.domain_classifier = nn.Linear(clip_model.visual.output_dim, domain_number, bias=False, dtype=self.dtype)

        self.logit_scale = clip_model.logit_scale # Use CLIP's learned scale for classification
        self.client_domain_id = client_domain_id
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temperature = contrastive_temp # Fixed temperature for contrastive loss

        print(f"FaST-PT Client initialized for Domain ID: {client_domain_id}")
        print(f"Contrastive Loss Lambda: {contrastive_lambda}, Temperature: {contrastive_temp}")


    def _calculate_logits(self, image_features, mode="composite", P_N=None):
        prompts, tokenized_prompts = self.prompt_learner(mode=mode, P_N=P_N)
        # print(f"prompts isnan: {torch.isnan(prompts).any()}, isinf: {torch.isinf(prompts).any()}")
        # if torch.isnan(prompts).any():
        #     print("prompts (from prompt_learner) CONTAINS NAN!")
        #     # 检查 self.prompt_learner.global_ctx 和 self.prompt_learner.domain_ctx 是否为 nan
        #     print(f"  global_ctx isnan: {torch.isnan(self.prompt_learner.global_ctx).any()}")
        #     print(f"  domain_ctx isnan: {torch.isnan(self.prompt_learner.domain_ctx).any()}")
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # # Normalize features
        # image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        # Normalize features
        epsilon = 1e-8 # 或者使用与 c_loss 中相同的 1e-6
        image_features_norm = image_features / (image_features.norm(dim=-1, keepdim=True) + epsilon)
        text_features_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + epsilon)

        # Match dtypes to avoid mismatch(added)
        dtype = text_features_norm.dtype
        image_features_norm = image_features_norm.to(dtype)

        # Calculate logits using CLIP's learned scale
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features_norm @ text_features_norm.t()
        return logits

    def obtain_prompt(self,P_D_list=None):
        prompt_state_dict = {
            'global_ctx': self.prompt_learner.global_ctx,
            'domain_classifier': self.domain_classifier.state_dict(),
            'P_D_list': P_D_list,
        }
        return prompt_state_dict

    def inference_step(self, feature, P_D_list, global_P_G):
        device = feature.device
        self.eval() # Set model to evaluation mode

        # Ensure parameters are on the correct device
        P_D_list = P_D_list.to(device).type(self.dtype)
        global_P_G = global_P_G.to(device).type(self.dtype)

        self.prompt_learner.global_ctx.data = global_P_G

        with torch.no_grad():

            domain_logits = self.domain_classifier(feature) # Shape: [Batch, K]
            domain_probs = F.softmax(domain_logits, dim=-1) # Shape: [Batch, K]

            #Generate dynamic prompt P^N via weighted sum (Eq. 9)
            P_N = torch.einsum('bk,knd->bnd', domain_probs, P_D_list) # Shape: [Batch, L_domain, Dim]

            batch_size = feature.shape[0]
            all_logits = []
            for i in range(batch_size):
                p_n_i = P_N[i] # Shape: [L_domain, Dim]
                logits_i = self._calculate_logits(feature[i:i+1], mode="inference", P_N=p_n_i)
                all_logits.append(logits_i)

            final_logits = torch.cat(all_logits, dim=0) # Shape: [Batch, N_Classes]

        return final_logits

def train_global(model,train_loader,aug_loader,optimizer,loss_fun,device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    batch_idx = 0
    for feature,label,domain in aug_loader:
        feature = feature.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # Forward pass
        logits_g_original = model._calculate_logits(feature, mode="global_only") # WORKAROUND
        # print(logits_g_original, label)
        loss = loss_fun(logits_g_original, label)
        loss_all += loss.item()
        total += label.size(0)
        _, predicted = logits_g_original.max(1)
        correct += predicted.eq(label).sum().item()
        # loss_isnan = torch.isnan(loss).any()
        # loss_isinf = torch.isinf(loss).any()
        # if loss_isnan or loss_isinf:
        #     print(f"Loss in train_global (batch {batch_idx}) isnan: {loss_isnan}, isinf: {loss_isinf}. Loss value: {loss.item() if not loss_isnan else 'NaN'}")
        # else:
        #     print(f"Loss in train_global (batch {batch_idx}): {loss.item()}")

        # if not loss_isnan and not loss_isinf:
        #     loss.backward()
        #     print(f"Called loss.backward() for batch {batch_idx}.")

        #     # 检查梯度
        #     if model.prompt_learner.global_ctx.grad is not None:
        #         grad_isnan = torch.isnan(model.prompt_learner.global_ctx.grad).any()
        #         grad_isinf = torch.isinf(model.prompt_learner.global_ctx.grad).any()
        #         if grad_isnan or grad_isinf:
        #             print(f"GRADIENT of global_ctx (batch {batch_idx}) isnan: {grad_isnan}, isinf: {grad_isinf}.")
        #         else:
        #             print(f"GRADIENT of global_ctx (batch {batch_idx}) is OK.")
                
        #         # --- 添加梯度裁剪 ---
        #         # 只在梯度有效时裁剪和更新
        #         if not grad_isnan and not grad_isinf:
        #              torch.nn.utils.clip_grad_norm_(model.prompt_learner.global_ctx, max_norm=1.0) # max_norm=1.0 是一个常用值
        #              print(f"Gradient CLIPPED for global_ctx in train_global (batch {batch_idx}).")
                     
        #              optimizer.step()
        #              print(f"Called optimizer.step() for batch {batch_idx}.")

        #              gc_isnan_after_step = torch.isnan(model.prompt_learner.global_ctx.data).any()
        #              gc_isinf_after_step = torch.isinf(model.prompt_learner.global_ctx.data).any()
        #              if gc_isnan_after_step or gc_isinf_after_step:
        #                 print(f"global_ctx (batch {batch_idx}) AFTER optimizer.step() isnan: {gc_isnan_after_step}, isinf: {gc_isinf_after_step}.")
        #              else:
        #                 print(f"global_ctx (batch {batch_idx}) is OK AFTER optimizer.step().")
        #         else:
        #             print(f"Skipping optimizer step for global_ctx due to NaN/Inf GRADIENT in train_global (batch {batch_idx}).")
        #     else:
        #         print(f"GRADIENT of global_ctx in train_global (batch {batch_idx}) is None.")
        # else:
        #     print(f"Skipping backward() and optimizer.step() in train_global (batch {batch_idx}) because LOSS is NaN or Inf.")

        # 检查 prompts 状态 (通过 global_ctx 间接检查) 在一个 batch 处理完后
        # Backward pass
        loss.backward()
        optimizer.step()
        batch_idx += 1
    return loss_all / len(aug_loader), correct / total


def train_domain(model,train_loader,aug_loader,optimizer,loss_fun,device,handcrafted_domain_prompt):
    model.train()
    loss_all = 0
    total = 0
    correct = 0

    for feature,label,domain in train_loader:
        feature = feature.half()
        feature = feature.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # Forward pass
        logits_g_original = model._calculate_logits(feature, mode="composite") # WORKAROUND
        # print(logits_g_original, label)
        loss = loss_fun(logits_g_original, label) + c_loss(model,handcrafted_domain_prompt,device)
        loss_all += loss.item()
        total += label.size(0)
        _, predicted = logits_g_original.max(1)
        correct += predicted.eq(label).sum().item()

        # Backward pass
        loss.backward()
        optimizer.step()
    return loss_all / len(train_loader), correct / total


def train_domclassifier(model,train_loader,aug_loader,optimizer,loss_fun,device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    model.to(device)

    for feature,label,domain in aug_loader:
        feature = feature.half()
        feature = feature.to(device)
        domain = domain.to(device)
        optimizer.zero_grad()
        # Forward pass
        domain_preds_original = model.domain_classifier(feature)
        loss = loss_fun(domain_preds_original, domain)
        loss_all += loss.item()
        total += domain.size(0)
        _, predicted = domain_preds_original.max(1)
        correct += predicted.eq(domain).sum().item()

        # Backward pass
        loss.backward()
        optimizer.step()
    return loss_all / len(train_loader), correct / total

def Aggregation(server_model,client_model,client_weights,device):
    num_clients = len(client_model)
    with torch.no_grad():

        pg_server_param = server_model.prompt_learner.global_ctx
        
        aggregated_pg = torch.zeros_like(pg_server_param.data, dtype=torch.float32).to(device)
        for i in range(num_clients):
            client_pg_param = client_model[i].prompt_learner.global_ctx
            # print(aggregated_pg.device,  client_pg_param.data.device)
            aggregated_pg += client_weights[i] * client_pg_param.data.float()

        server_model.prompt_learner.global_ctx.data.copy_(aggregated_pg.type_as(pg_server_param.data))

        for i in range(num_clients):
            client_model[i].prompt_learner.global_ctx.data.copy_(aggregated_pg.type_as(pg_server_param.data))

        dc_weight_server_param = server_model.domain_classifier.weight
        aggregated_dc_weight = torch.zeros_like(dc_weight_server_param.data, dtype=torch.float32).to(device)

        for i in range(num_clients):
            client_dc_weight_param = client_model[i].domain_classifier.weight
            aggregated_dc_weight += client_weights[i] * client_dc_weight_param.data.float()

        server_model.domain_classifier.weight.data.copy_(aggregated_dc_weight.type_as(dc_weight_server_param.data))

        for i in range(num_clients):
            client_model[i].domain_classifier.weight.data.copy_(aggregated_dc_weight.type_as(dc_weight_server_param.data))

        if server_model.domain_classifier.bias is not None:
            dc_bias_server_param = server_model.domain_classifier.bias
            aggregated_dc_bias = torch.zeros_like(dc_bias_server_param.data, dtype=torch.float32)

            for i in range(num_clients):
                client_dc_bias_param = client_model[i].domain_classifier.bias
                if client_dc_bias_param is not None: 
                    aggregated_dc_bias += client_weights[i] * client_dc_bias_param.data.float()

            server_model.domain_classifier.bias.data.copy_(aggregated_dc_bias.type_as(dc_bias_server_param.data))

            for i in range(num_clients):
                if client_model[i].domain_classifier.bias is not None:
                    client_model[i].domain_classifier.bias.data.copy_(aggregated_dc_bias.type_as(dc_bias_server_param.data))
        

def test_infer(model,data_loader,device,P_D_list):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for feature,label,domain in data_loader:
            feature = feature.half()
            feature = feature.to(device)
            label = label.to(device)
            final_logits = model.inference_step(feature,P_D_list,model.prompt_learner.global_ctx)
            _, predicted = final_logits.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
    return correct / total


