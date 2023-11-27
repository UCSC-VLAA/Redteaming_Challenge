# Copyright (c) 2023 Andy Zou https://github.com/llm-attacks/llm-attacks. All rights reserved.
# This file has been modified by Zijun Wang ("Zijun Wang Modifications").
# All Zijun Wang Modifications are Copyright (C) 2023 Zijun Wang. All rights reserved.

import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import json
from filelock import FileLock

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings, get_controls


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    return one_hot.grad.clone()


class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model, mode="control"):
        if mode == "control":
            return token_gradients(
                model, 
                self.input_ids.to(model.device), 
                self._control_slice, 
                self._target_slice, 
                self._loss_slice
            )
        elif mode == "prefix":
            return token_gradients(
                model, 
                self.input_ids.to(model.device), 
                self._prefix_slice, 
                self._target_slice, 
                self._loss_slice
            )
        else:
            raise ValueError(f"Invalid mode type, should be 'control' or 'prefix', got {mode}")


class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, mode, batch_size, topk=256, temp=1, allow_non_ascii=True):

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices
        
        if mode == "control":
            control_toks = self.control_toks.to(grad.device)
        elif mode == "prefix":
            control_toks = self.prefix_toks.to(grad.device)
        else:
            raise ValueError(f"Invalid mode type, should be 'control' or 'prefix', got {mode}")
        
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        
        return new_control_toks


class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
    
    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1.0, 
             control_weight=0.1, 
             verbose=False, 
             opt_only=False,
             change_prefix=False,
             enable_prefix_sharing=True,
             merge_loss_compute=True,
             skip_model_worker=True,
             filter_cand=True,
             choose_control_prob=0.5):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        if merge_loss_compute:
            assert target_weight > 0
            loss_config = [(target_weight, "target_loss"), ]
            if control_weight > 0:
                loss_config.append((control_weight, "control_loss"))
        else:
            loss_config = None

        main_device = self.models[0].device
        control_candses = []
        losses = []
        
        curr_control_dict = {
            'control': self.control_str,
            'prefix': self.prefix_str
        }
        import time
        start = time.time()
        
        if change_prefix:
            modes = ["control"]
        else:
            modes = ["control", "prefix"]
            p = [choose_control_prob, 1-choose_control_prob]
            modes = [modes[np.random.choice(len(modes), p=p)]]

        for mode in modes:
            control_cands = []
            curr_control = curr_control_dict[mode]

            if not skip_model_worker:
                for j, worker in enumerate(self.workers):
                    worker(self.prompts[j], "grad", worker.model, mode)
            else:
                with torch.enable_grad():
                    grads_skip_model_worker = [self.prompts[j].grad(worker.model, mode) for j, worker in enumerate(self.workers)]

            # Aggregate gradients
            grad = None
            for j, worker in enumerate(self.workers):
                new_grad = worker.results.get().to(main_device) if not skip_model_worker else grads_skip_model_worker[j]
                new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
                if grad is None:
                    grad = torch.zeros_like(new_grad)
                if grad.shape != new_grad.shape:
                    with torch.no_grad():
                        control_cand = self.prompts[j-1].sample_control(grad, mode, batch_size, topk, temp, allow_non_ascii)
                        control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=curr_control))
                    grad = new_grad
                else:
                    grad += new_grad
                with torch.no_grad():
                    control_cand = self.prompts[j].sample_control(grad, mode, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=curr_control))
                    
            del grad, control_cand ; gc.collect()
            control_candses.append(control_cands)

            # Search
            loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
            with torch.no_grad():
                for j, cand in enumerate(control_cands):
                    progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else range(len(
                        self.prompts[0]))
                    for i in progress:
                        if not skip_model_worker:
                            for k, worker in enumerate(self.workers):
                                worker(self.prompts[k][i],
                                       "logits",
                                       worker.model,
                                       cand,
                                       mode,
                                       return_ids=True,
                                       enable_prefix_sharing=enable_prefix_sharing,
                                       loss_config=loss_config)

                            if merge_loss_compute:
                                _, _, loss_infos = zip(*[worker.results.get() for worker in self.workers])

                                for loss_info in loss_infos:
                                    for _, (weight, loss_i) in loss_info.items():
                                        loss[j * batch_size:(j + 1) * batch_size] += weight * loss_i.mean(dim=-1).to(main_device)
                            else:
                                logits, ids = zip(*[worker.results.get() for worker in self.workers])
                                loss[j*batch_size:(j+1)*batch_size] += sum([
                                    target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device)
                                    for k, (logit, id) in enumerate(zip(logits, ids))
                                ])
                                if control_weight != 0:
                                    loss[j*batch_size:(j+1)*batch_size] += sum([
                                        control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                                        for k, (logit, id) in enumerate(zip(logits, ids))
                                    ])
                                del logits, ids ; gc.collect()
                        else:
                            assert merge_loss_compute
                            for k, worker in enumerate(self.workers):
                                _, _, loss_info = self.prompts[k][i].logits(
                                    worker.model,
                                    cand,
                                    mode,
                                    return_ids=True,
                                    enable_prefix_sharing=enable_prefix_sharing,
                                    loss_config=loss_config
                                )
                                for _, (weight, loss_i) in loss_info.items():
                                    loss[j * batch_size:(j + 1) * batch_size] += weight * loss_i.mean(dim=-1).to(main_device)

                        if verbose:
                            progress.set_description(f"loss_{mode}={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")
            
            losses.append(loss)
        
        if change_prefix:
            prefix_bs = batch_size // 2
            loss_prefix = torch.zeros(prefix_bs).to(main_device)
            
            with FileLock(self.prefix_dataset_path+".lock"):
                with open(self.prefix_dataset_path, 'r') as f:
                    prefix_dataset = json.load(f)
            prefixs = get_controls(prefix_dataset, prefix_bs)          
            progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else range(len(self.prompts[0]))
            for i in progress:
                if not skip_model_worker:
                    for k, worker in enumerate(self.workers):
                        worker(
                            self.prompts[k][i],
                            "logits",
                            worker.model,
                            prefixs,
                            "prefix",
                            return_ids=True,
                            enable_prefix_sharing=enable_prefix_sharing,
                            loss_config=loss_config
                        )
                    if merge_loss_compute:
                        _, _, loss_infos = zip(*[worker.results.get() for worker in self.workers])

                        for loss_info in loss_infos:
                            for _, (weight, loss_i) in loss_info.items():
                                loss_prefix[0:prefix_bs] += weight * loss_i.mean(dim=-1).to(main_device)
                    else:
                        logits, ids = zip(*[worker.results.get() for worker in self.workers])
                        loss_prefix[0:prefix_bs] += sum([
                            target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                        if control_weight != 0:
                            loss_prefix[0:prefix_bs] += sum([
                                control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                                for k, (logit, id) in enumerate(zip(logits, ids))
                            ])
                        del logits, ids ; gc.collect()
                else:
                    assert merge_loss_compute
                    for k, worker in enumerate(self.workers):
                        _, _, loss_info = self.prompts[k][i].logits(
                            worker.model,
                            prefixs,
                            "prefix",
                            return_ids=True,
                            enable_prefix_sharing=enable_prefix_sharing,
                            loss_config=loss_config
                        )
                        for _, (weight, loss_i) in loss_info.items():
                            loss_prefix[0:prefix_bs] += weight * loss_i.mean(dim=-1).to(main_device)

                if verbose:
                    progress.set_description(f"loss_prefix_whole={loss_prefix.min().item()/(i+1):.4f}")
            losses.append(loss_prefix)
            modes.append("prefix_whole")
          
            
        min_loss = torch.tensor([loss.min() for loss in losses])
        if modes[min_loss.argmin()] == "control":
            min_idx = losses[min_loss.argmin()].argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_candses[min_loss.argmin()][model_idx][batch_idx], losses[min_loss.argmin()][min_idx]
            next_prefix = self.prefix_str
            print('Current control length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
            print(f"next_control: {next_control}")
        elif modes[min_loss.argmin()] == "prefix":
            min_idx = losses[min_loss.argmin()].argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_prefix, cand_loss = control_candses[min_loss.argmin()][model_idx][batch_idx], losses[min_loss.argmin()][min_idx]
            next_control = self.control_str
            print('Current prefix length:', len(self.workers[0].tokenizer(next_prefix).input_ids[1:]))
            print(f"next_prefix: {next_prefix}")
        elif modes[min_loss.argmin()] == "prefix_whole":
            min_idx = losses[min_loss.argmin()].argmin()
            next_prefix = prefixs[min_idx]
            cand_loss = losses[min_loss.argmin()][min_idx]
            next_control = self.control_str
            print('Current prefix length:', len(self.workers[0].tokenizer(next_prefix).input_ids[1:]))
            print(f"next_prefix_whole: {next_prefix}")
        
        torch.cuda.synchronize()
        print(f'Step time: {time.time()-start}, loss_min: {min_loss.cpu().numpy().tolist()}')
        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers), next_prefix
