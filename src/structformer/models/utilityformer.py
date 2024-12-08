import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from structformer.models.object_selection_network import ObjectSelectionWUtility
from structformer.models.pose_generation_network import UtilityFormerPoseGenerationWUtility

class UtilityFomrer(nn.Module):
    def __init__(self, vocab_size, model_dim=256, obj_selection_cfg: dict = None, pos_generation_cfg: dict = None):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        obj_selection_cfg['model_dim'] = model_dim
        pos_generation_cfg['model_dim'] = model_dim

        self.obj_selection_model = ObjectSelectionWUtility(**obj_selection_cfg)
        self.pos_generation_model = UtilityFormerPoseGenerationWUtility(**pos_generation_cfg)

    
    def forward(self, xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                sentence, sentence_pad_mask, token_type_index,
                obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index, tgt_mask, start_token,
                struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                struct_position_index, struct_token_type_index, struct_pad_mask):
        batch_size, _ = sentence_pad_mask.size()

        max_obj = xyzs.size(0) // batch_size
        max_other_obj = other_xyzs.size(0) // batch_size
        sel_input_dim = batch_size * (max_obj + max_other_obj)
        sel_num_points = xyzs.size(1)

        # get utility
        word_embed = self.word_embeddings(sentence)
        # in args pad 0 means valid
        word_pad_mask = 1 - sentence_pad_mask.reshape(batch_size, -1, 1).float()
        word_embed = word_embed * word_pad_mask

        # sum as utility embedding
        utility_embed = torch.sum(word_embed, dim=1, keepdim=True)

        all_object_xyzs = torch.cat([xyzs.reshape(batch_size, max_obj, -1), other_xyzs.reshape(batch_size, max_other_obj, -1)], dim=1).reshape(sel_input_dim, sel_num_points, -1)
        all_object_rgbs = torch.cat([rgbs.reshape(batch_size, max_obj, -1), other_rgbs.reshape(batch_size, max_other_obj, -1)], dim=1).reshape(sel_input_dim, sel_num_points, -1)
        all_object_pad_mask = torch.cat([object_pad_mask, other_object_pad_mask], dim=1)

        selection_result = self.obj_selection_model.forward(all_object_xyzs, all_object_rgbs, all_object_pad_mask, utility_embed)
        pos_generation_result = self.pos_generation_model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                                                                    utility_embed, token_type_index,
                                                                    obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index, tgt_mask, start_token,
                                                                    struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                                                                    struct_position_index, struct_token_type_index, struct_pad_mask)
        
        return selection_result, pos_generation_result
        
    def criterion_obj(self, predictions, labels):
        return self.obj_selection_model.criterion(predictions, labels)

    def criterion_pos(self, predictions, labels):
        return self.pos_generation_model.criterion(predictions, labels)