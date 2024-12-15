from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import tqdm

import pickle
import argparse
from omegaconf import OmegaConf
from collections import defaultdict

from torch.utils.data import DataLoader
from structformer.data.utility_dataset import SequenceDataset,PreferenceSequenceDataset
from structformer.models.pose_generation_network import UtilityFormerPoseGenerationWSentence
from structformer.models.utilityformer import UtilityFomrer,UtilityFomrerUtilityOnly
from structformer.data.tokenizer import Tokenizer
from structformer.utils.rearrangement import evaluate_prior_prediction, generate_square_subsequent_mask
from structformer.training.train_object_selection_network import evaluate

from rich import print
from typing import Tuple

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """  
    # policy_chosen_logps: 训练模型对于chosen经过log后logits
    # policy_rejected_logps: 训练模型对于rejected经过log后logits
    # reference_chosen_logps: 训练模型对于chosen经过log后logits
    # reference_rejected_logps: 训练模型对于rejected经过log后logits
    # beta: policy和reference的差异性控制参数
    """
        
    # actor模型选择chosen优先于rejected
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    # reference模型选择chosen优先于rejected
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0
        
    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  
    else:
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        
    # chosen和rejected的奖励
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards





def train_model(cfg, model: UtilityFomrer, data_iter, optimizer, warmup, num_epochs, device, save_best_model, grad_clipping=1.0):

    if save_best_model:
        best_model_dir = os.path.join(cfg.experiment_dir, "best_model")
        print("best model will be saved to {}".format(best_model_dir))
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_score = -np.inf

    for epoch in range(num_epochs):
        print('''+================================================================+
| ____  _             _     _____          _       _             |
|/ ___|| |_ __ _ _ __| |_  |_   _| __ __ _(_)_ __ (_)_ __   __ _ |
|\___ \| __/ _` | '__| __|   | || '__/ _` | | '_ \| | '_ \ / _` ||
| ___) | || (_| | |  | |_    | || | | (_| | | | | | | | | | (_| ||
||____/ \__\__,_|_|   \__|   |_||_|  \__,_|_|_| |_|_|_| |_|\__, ||
|                                                          |___/ |
+================================================================+''')
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        epoch_loss = 0
        pos_gts = defaultdict(list)
        pos_predictions = defaultdict(list)
        obj_gts = defaultdict(list)
        obj_predictions = defaultdict(list)

        with tqdm.tqdm(total=len(data_iter["train"])) as pbar:
            for step, batch in enumerate(data_iter["train"]):
                optimizer.zero_grad()
                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                rgbs = batch["rgbs"].to(device, non_blocking=True)
                object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
                other_xyzs = batch["other_xyzs"].to(device, non_blocking=True)
                other_rgbs = batch["other_rgbs"].to(device, non_blocking=True)
                other_object_pad_mask = batch["other_object_pad_mask"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)
                obj_x_inputs = batch["obj_x_inputs"].to(device, non_blocking=True)
                obj_y_inputs = batch["obj_y_inputs"].to(device, non_blocking=True)
                obj_z_inputs = batch["obj_z_inputs"].to(device, non_blocking=True)
                obj_theta_inputs = batch["obj_theta_inputs"].to(device, non_blocking=True)

                struct_x_inputs = batch["struct_x_inputs"].to(device, non_blocking=True)
                struct_y_inputs = batch["struct_y_inputs"].to(device, non_blocking=True)
                struct_z_inputs = batch["struct_z_inputs"].to(device, non_blocking=True)
                struct_theta_inputs = batch["struct_theta_inputs"].to(device, non_blocking=True)
                struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
                struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
                struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

                tgt_mask = generate_square_subsequent_mask(object_pad_mask.shape[1] + 1).to(device, non_blocking=True)
                start_token = torch.zeros((object_pad_mask.shape[0], 1), dtype=torch.long).to(device, non_blocking=True)

                # output
                pos_targets = {}
                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    pos_targets[key] = batch[key].to(device, non_blocking=True)
                    pos_targets[key] = pos_targets[key].reshape(pos_targets[key].shape[0] * pos_targets[key].shape[1], -1)

                obj_targets = {}
                # check datasets
                obj_targets['rearrange_obj_labels'] = torch.cat([1 - object_pad_mask.float() + object_pad_mask.float() * -100.0,
                                                                 other_object_pad_mask.float() * 0 + other_object_pad_mask.float() * -100.0], dim=1).reshape(-1)

                (obj_preds, pos_preds) = model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                                      sentence, sentence_pad_mask, token_type_index,
                                      obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index,
                                      tgt_mask, start_token,
                                      struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                                      struct_position_index, struct_token_type_index, struct_pad_mask)

                loss_pos = model.criterion_pos(pos_preds, pos_targets)
                loss_obj = model.criterion_obj(obj_preds, obj_targets)

                loss = loss_pos + loss_obj
                loss.backward()

                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

                optimizer.step()

                loss = loss.item()
                epoch_loss += loss

                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    pos_gts[key].append(pos_targets[key].detach())
                    pos_predictions[key].append(pos_preds[key].detach())

                for key in ["rearrange_obj_labels"]:
                    obj_gts[key].append(obj_targets[key].detach())
                    obj_predictions[key].append(obj_preds[key].detach())

                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss, 'Pos loss': loss_pos.item(), 'Obj loss': loss_obj.item()})

        warmup.step()

        print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, epoch_loss))
        print('''+==============================================+
| ____   ___  ____    _______     ___    _     |
||  _ \ / _ \/ ___|  | ____\ \   / / \  | |    |
|| |_) | | | \___ \  |  _|  \ \ / / _ \ | |    |
||  __/| |_| |___) | | |___  \ V / ___ \| |___ |
||_|    \___/|____/  |_____|  \_/_/   \_\_____||
+==============================================+''')
        evaluate_prior_prediction(pos_gts, pos_predictions, ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                                    "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"])

        print('''+==============================================+
|  ___  ____      _   _______     ___    _     |
| / _ \| __ )    | | | ____\ \   / / \  | |    |
|| | | |  _ \ _  | | |  _|  \ \ / / _ \ | |    |
|| |_| | |_) | |_| | | |___  \ V / ___ \| |___ |
| \___/|____/ \___/  |_____|  \_/_/   \_\_____||
+==============================================+''')
        evaluate(obj_gts, obj_predictions, ["rearrange_obj_labels"])
        
        score = validate(cfg, model, data_iter["valid"], epoch, device)
        if save_best_model and score > best_score:
            print("Saving best model so far...")
            best_score = score
            save_model(best_model_dir, cfg, epoch, model)

    return model


@torch.no_grad()
def validate(cfg, model: UtilityFomrer, data_iter, epoch, device):
    """
    helper function to evaluate the model

    :param model:
    :param data_iter:
    :param epoch:
    :param device:
    :return:
    """
    print('''+=========================================================================+
| ____  _             _    __     __    _ _     _       _   _             |
|/ ___|| |_ __ _ _ __| |_  \ \   / /_ _| (_) __| | __ _| |_(_) ___  _ __  |
|\___ \| __/ _` | '__| __|  \ \ / / _` | | |/ _` |/ _` | __| |/ _ \| '_ \ |
| ___) | || (_| | |  | |_    \ V / (_| | | | (_| | (_| | |_| | (_) | | | ||
||____/ \__\__,_|_|   \__|    \_/ \__,_|_|_|\__,_|\__,_|\__|_|\___/|_| |_||
+=========================================================================+''')

    model.eval()
    epoch_loss = 0
    pos_gts = defaultdict(list)
    pos_predictions = defaultdict(list)
    obj_gts = defaultdict(list)
    obj_predictions = defaultdict(list)

    with tqdm.tqdm(total=len(data_iter)) as pbar:
        for step, batch in enumerate(data_iter):
            # input
            xyzs = batch["xyzs"].to(device, non_blocking=True)
            rgbs = batch["rgbs"].to(device, non_blocking=True)
            object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
            other_xyzs = batch["other_xyzs"].to(device, non_blocking=True)
            other_rgbs = batch["other_rgbs"].to(device, non_blocking=True)
            other_object_pad_mask = batch["other_object_pad_mask"].to(device, non_blocking=True)
            sentence = batch["sentence"].to(device, non_blocking=True)
            sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
            token_type_index = batch["token_type_index"].to(device, non_blocking=True)
            position_index = batch["position_index"].to(device, non_blocking=True)
            obj_x_inputs = batch["obj_x_inputs"].to(device, non_blocking=True)
            obj_y_inputs = batch["obj_y_inputs"].to(device, non_blocking=True)
            obj_z_inputs = batch["obj_z_inputs"].to(device, non_blocking=True)
            obj_theta_inputs = batch["obj_theta_inputs"].to(device, non_blocking=True)

            struct_x_inputs = batch["struct_x_inputs"].to(device, non_blocking=True)
            struct_y_inputs = batch["struct_y_inputs"].to(device, non_blocking=True)
            struct_z_inputs = batch["struct_z_inputs"].to(device, non_blocking=True)
            struct_theta_inputs = batch["struct_theta_inputs"].to(device, non_blocking=True)
            struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
            struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
            struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

            tgt_mask = generate_square_subsequent_mask(object_pad_mask.shape[1] + 1).to(device, non_blocking=True)
            start_token = torch.zeros((object_pad_mask.shape[0], 1), dtype=torch.long).to(device, non_blocking=True)

            # output
            pos_targets = {}
            for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                        "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                pos_targets[key] = batch[key].to(device, non_blocking=True)
                pos_targets[key] = pos_targets[key].reshape(pos_targets[key].shape[0] * pos_targets[key].shape[1], -1)

            obj_targets = {}
            # check datasets
            obj_targets['rearrange_obj_labels'] = torch.cat([1 - object_pad_mask.float() + object_pad_mask.float() * -100.0,
                 other_object_pad_mask.float() * 0 + other_object_pad_mask.float() * -100.0], dim=1).reshape(-1)

            (obj_preds, pos_preds) = model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                                    sentence, sentence_pad_mask, token_type_index,
                                    obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index,
                                    tgt_mask, start_token,
                                    struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                                    struct_position_index, struct_token_type_index, struct_pad_mask)

            loss_pos = model.criterion_pos(pos_preds, pos_targets)
            loss_obj = model.criterion_obj(obj_preds, obj_targets)

            loss = loss_pos + loss_obj
            loss = loss.item()
            epoch_loss += loss

            for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                        "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                pos_gts[key].append(pos_targets[key].detach())
                pos_predictions[key].append(pos_preds[key].detach())

            for key in ["rearrange_obj_labels"]:
                obj_gts[key].append(obj_targets[key].detach())
                obj_predictions[key].append(obj_preds[key].detach())

            pbar.update(1)
            pbar.set_postfix({"Batch loss": loss, 'Pos loss': loss_pos.item(), 'Obj loss': loss_obj.item()})

    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, epoch_loss))
    print('''+==============================================+
| ____   ___  ____    _______     ___    _     |
||  _ \ / _ \/ ___|  | ____\ \   / / \  | |    |
|| |_) | | | \___ \  |  _|  \ \ / / _ \ | |    |
||  __/| |_| |___) | | |___  \ V / ___ \| |___ |
||_|    \___/|____/  |_____|  \_/_/   \_\_____||
+==============================================+''')
    pos_score = evaluate_prior_prediction(pos_gts, pos_predictions, ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                                "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"])

    print('''+==============================================+
|  ___  ____      _   _______     ___    _     |
| / _ \| __ )    | | | ____\ \   / / \  | |    |
|| | | |  _ \ _  | | |  _|  \ \ / / _ \ | |    |
|| |_| | |_) | |_| | | |___  \ V / ___ \| |___ |
| \___/|____/ \___/  |_____|  \_/_/   \_\_____||
+==============================================+''')
    obj_score = evaluate(obj_gts, obj_predictions, ["rearrange_obj_labels"])
    
    return pos_score + obj_score


# TODO modify this to utilityformer version
def infer_once(cfg, model, batch, device):

    model.eval()

    predictions = defaultdict(list)
    with torch.no_grad():

        # input
        batch_size = batch.get("batch_size")
        xyzs = batch["xyzs"].to(device, non_blocking=True)
        rgbs = batch["rgbs"].to(device, non_blocking=True)
        object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
        other_xyzs = batch["other_xyzs"].to(device, non_blocking=True)
        other_rgbs = batch["other_rgbs"].to(device, non_blocking=True)
        other_object_pad_mask = batch["other_object_pad_mask"].to(device, non_blocking=True)
        sentence = None
        sentence_pad_mask =None
        if "sentence" in batch:
            sentence = batch["sentence"].to(device, non_blocking=True)
            sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
        token_type_index = batch["token_type_index"].to(device, non_blocking=True)
        position_index = batch["position_index"].to(device, non_blocking=True)

        struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
        struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
        struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

        obj_x_inputs = batch["obj_x_inputs"].to(device, non_blocking=True)
        obj_y_inputs = batch["obj_y_inputs"].to(device, non_blocking=True)
        obj_z_inputs = batch["obj_z_inputs"].to(device, non_blocking=True)
        obj_theta_inputs = batch["obj_theta_inputs"].to(device, non_blocking=True)

        struct_x_inputs = batch["struct_x_inputs"].to(device, non_blocking=True)
        struct_y_inputs = batch["struct_y_inputs"].to(device, non_blocking=True)
        struct_z_inputs = batch["struct_z_inputs"].to(device, non_blocking=True)
        struct_theta_inputs = batch["struct_theta_inputs"].to(device, non_blocking=True)

        tgt_mask = generate_square_subsequent_mask(object_pad_mask.shape[1] + 1).to(device, non_blocking=True)
        start_token = torch.zeros((object_pad_mask.shape[0], 1), dtype=torch.long).to(device, non_blocking=True)

        (obj_preds, pos_preds) = model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                                    sentence, sentence_pad_mask, token_type_index,
                                    obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index,
                                    tgt_mask, start_token,
                                    struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                                    struct_position_index, struct_token_type_index, struct_pad_mask,
                                    batch_size=batch_size)

        for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                    "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
            predictions[key].append(pos_preds[key])

    return predictions


def save_model(model_dir, cfg, epoch, model, optimizer=None, scheduler=None):
    print(f"save_model {epoch}")
    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state_dict, os.path.join(model_dir, "model.tar"))
    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_model(model_dir, dirs_cfg):
    """
    Load transformer model
    Important: to use the model, call model.eval() or model.train()
    :param model_dir:
    :return:
    """
    # load dictionaries
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    if dirs_cfg:
        cfg = OmegaConf.merge(cfg, dirs_cfg)

    data_cfg = cfg.dataset
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    # initialize model
    model_cfg = cfg.model
    obj_selections_cfg = model_cfg.obj_selection
    pos_generation_cfg = model_cfg.pos_generation
    model = UtilityFomrerUtilityOnly(vocab_size, model_cfg.model_dim, obj_selections_cfg, pos_generation_cfg)

    model.to(cfg.device)
    checkpoint = torch.load(os.path.join(model_dir, "model.tar"))
    model = load_model_from_pretrained(model,model_dir)
    
    optimizer = None
    if "optimizer_state_dict" in checkpoint:
        training_cfg = cfg.training
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if "scheduler_state_dict" in checkpoint:
        scheduler = None
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    return cfg, tokenizer, model, optimizer, scheduler, epoch

def load_model_from_pretrained(model:UtilityFomrer,model_dir:str):
    checkpoint = torch.load(os.path.join(model_dir, "model.tar"))
    model_names = [name for name in checkpoint["model_state_dict"].keys()]
    # if there are no utility_embeddings
    if hasattr(model,"utility_embeddings") and "utility_embeddings" not in model_names:
        word_embeddings = checkpoint["model_state_dict"]['word_embeddings.weight']
        utility_embeddings = torch.mean(word_embeddings, dim=0, keepdim=True)*(word_embeddings.size(0)//10)
        checkpoint["model_state_dict"].update({'utility_embeddings.weight':utility_embeddings})
    # load state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def run_model(cfg):

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True

    data_cfg = cfg.dataset

    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    train_dataset = PreferenceSequenceDataset(data_cfg.dirs, data_cfg.index_dirs, "test", tokenizer,
                                    data_cfg.max_num_objects,
                                    data_cfg.max_num_other_objects,
                                    data_cfg.max_num_shape_parameters,
                                    data_cfg.max_num_rearrange_features,
                                    data_cfg.max_num_anchor_features,
                                    data_cfg.num_pts,
                                    data_cfg.use_structure_frame,
                                    filter=data_cfg.filter)
    valid_dataset = PreferenceSequenceDataset(data_cfg.dirs, data_cfg.index_dirs, "test", tokenizer,
                                    data_cfg.max_num_objects,
                                    data_cfg.max_num_other_objects,
                                    data_cfg.max_num_shape_parameters,
                                    data_cfg.max_num_rearrange_features,
                                    data_cfg.max_num_anchor_features,
                                    data_cfg.num_pts,
                                    data_cfg.use_structure_frame,
                                    filter = data_cfg.filter)

    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                    collate_fn=PreferenceSequenceDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers, persistent_workers=True)
    data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    collate_fn=PreferenceSequenceDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers, persistent_workers=True)
    
    # load model
    model_cfg = cfg.model
    obj_selections_cfg = model_cfg.obj_selection
    pos_generation_cfg = model_cfg.pos_generation
    model = UtilityFomrerUtilityOnly(vocab_size, model_cfg.model_dim, obj_selections_cfg, pos_generation_cfg)
    model.to(cfg.device)
    model = load_model_from_pretrained(model,model_dir=model_cfg.pretrained_model)
    
    for name, parameter in model.named_parameters():
        if "utility_embeddings" not in name:
            parameter.requires_grad = False


    training_cfg = cfg.training
    optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=training_cfg.learning_rate, weight_decay=training_cfg.l2, amsgrad=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.lr_restart)
    warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=training_cfg.warmup,
                                    after_scheduler=scheduler)

    train_model(cfg, model, data_iter, optimizer, warmup, training_cfg.max_epochs, cfg.device, cfg.save_best_model)

    # save model
    if cfg.save_model:
        model_dir = os.path.join(cfg.experiment_dir, "model")
        print("Saving model to {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model(model_dir, cfg, cfg.training.max_epochs, model, optimizer, scheduler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", help='location of the dataset', type=str)
    parser.add_argument("--main_config", help='config yaml file for the model',
                        default='configs/utilityformer_preference.yaml',
                        type=str)
    parser.add_argument("--dirs_config", help='config yaml file for directories',
                        default='configs/data/preference_circle.yaml',
                        type=str)
    args = parser.parse_args()

    # # debug
    # args.dataset_base_dir = "/home/weiyu/data_drive/data_new_objects"

    assert os.path.exists(args.main_config), "Cannot find config yaml file at {}".format(args.main_config)
    assert os.path.exists(args.dirs_config), "Cannot find config yaml file at {}".format(args.dir_config)

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")

    main_cfg = OmegaConf.load(args.main_config)
    dirs_cfg = OmegaConf.load(args.dirs_config)

    cfg = OmegaConf.merge(main_cfg, dirs_cfg)

    OmegaConf.resolve(cfg)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    run_model(cfg)