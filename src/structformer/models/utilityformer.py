
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from structformer.models.object_selection_network import ObjectSelectionWUtility,EncoderMLP
from structformer.models.pose_generation_network import UtilityFormerPoseGenerationWUtility
from structformer.models.point_transformer import PointTransformerEncoderSmall


class UtilityFomrer(nn.Module):
    def __init__(self, vocab_size, model_dim=256,obj_selection_cfg: dict = None, pos_generation_cfg: dict = None):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        self.utility_embeddings = nn.Embedding(1, model_dim)
        obj_selection_cfg['model_dim'] = model_dim
        pos_generation_cfg['model_dim'] = model_dim

        self.obj_selection_model = ObjectSelectionWUtility(**obj_selection_cfg)
        self.pos_generation_model = UtilityFormerPoseGenerationWUtility(**pos_generation_cfg)
  
    def forward(self, xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                sentence,sentence_pad_mask,token_type_index,
                obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index, tgt_mask, start_token,
                struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                struct_position_index, struct_token_type_index, struct_pad_mask,
                batch_size=None,):
        
        if sentence_pad_mask is not None:
            batch_size, _ = sentence_pad_mask.size()
        device = xyzs.device

        max_obj = xyzs.size(0) // batch_size
        max_other_obj = other_xyzs.size(0) // batch_size
        sel_input_dim = batch_size * (max_obj + max_other_obj)
        sel_num_points = xyzs.size(1)

        # get utility   
        utility_inputs = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        utility_embed = self.utility_embeddings(utility_inputs)
        if sentence is not None:
            word_embed = self.word_embeddings(sentence)
            # in args pad 0 means valid
            word_pad_mask = 1 - sentence_pad_mask.reshape(batch_size, -1, 1).float()
            word_embed = word_embed * word_pad_mask

            # sum as utility embedding
            word_sum_embed = torch.sum(word_embed, dim=1, keepdim=True)
            # avg pooling
            utility_embed = word_sum_embed

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
    
    
class UtilityFomrerUtilityOnly(UtilityFomrer):
    def __init__(self, vocab_size, model_dim=256,obj_selection_cfg: dict = None, pos_generation_cfg: dict = None):
        super().__init__(vocab_size, model_dim,obj_selection_cfg, pos_generation_cfg)

    
    def forward(self, xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                sentence,sentence_pad_mask,token_type_index,
                obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index, tgt_mask, start_token,
                struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                struct_position_index, struct_token_type_index, struct_pad_mask,
                batch_size=None,):
        if sentence_pad_mask is not None:
            batch_size, _ = sentence_pad_mask.size()
        device = xyzs.device

        max_obj = xyzs.size(0) // batch_size
        max_other_obj = other_xyzs.size(0) // batch_size
        sel_input_dim = batch_size * (max_obj + max_other_obj)
        sel_num_points = xyzs.size(1)

        # get utility   
        utility_inputs = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        utility_embed = self.utility_embeddings(utility_inputs)

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
    
    
class DistillEncoder(nn.Module):
    def __init__(self, model_dim=256, output_dim=240, ignore_rgb=False):
        super().__init__()  # 初始化基类
        self.model_dim = model_dim
        input_dim = 3 if ignore_rgb else 6
        self.ignore_rgb = ignore_rgb
        self.object_encoder = PointTransformerEncoderSmall(output_dim=model_dim, input_dim=input_dim, mean_center=True)
        self.object_mlp = EncoderMLP(model_dim, output_dim, uses_pt=True)
        self.distill_mlp = nn.Sequential(
            nn.Linear(output_dim*2, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim),
        )
    
    def forward(self,origin_xyzs,origin_rgbs,transform_xyzs,transform_rgbs):
        # batch_size, num_objects, embedding_size
        batch_size = origin_xyzs.shape[0]
        origin_obj_pc_embed = self.encode_pc(origin_xyzs,origin_rgbs,batch_size,origin_xyzs.shape[1])
        transform_obj_pc_embed = self.encode_pc(transform_xyzs,transform_rgbs,batch_size,transform_xyzs.shape[1])
        
        # TODO 沿num_objects合并
        origin_embed = origin_obj_pc_embed.sum(dim=1)
        transform_embed = transform_obj_pc_embed.sum(dim=1)
        origin_embed = torch.unsqueeze(origin_embed,dim=1)
        transform_embed = torch.unsqueeze(transform_embed,dim=1)
        
        total_embed = torch.stack([origin_embed,transform_embed],dim=1)
        total_embed = total_embed.reshape(batch_size,-1)
        x = self.distill_mlp(total_embed)
        return x
    
    def encode_pc(self, xyzs, rgbs, batch_size, num_objects):
        xyzs = torch.reshape(xyzs,(-1,xyzs.shape[-2],xyzs.shape[-1]))
        if self.ignore_rgb:
            center_xyz, x = self.object_encoder(xyzs, None)
        else:
            rgbs = torch.reshape(rgbs,(-1,rgbs.shape[-2],rgbs.shape[-1]))
            center_xyz, x = self.object_encoder(xyzs, rgbs)
        obj_pc_embed = self.object_mlp(x, center_xyz)
        obj_pc_embed = torch.reshape(obj_pc_embed,(batch_size, num_objects, -1))
        return obj_pc_embed
    
class UtilityFomrerWDistill(UtilityFomrer):
    def __init__(self, vocab_size, model_dim=256, obj_selection_cfg = None, pos_generation_cfg = None):
        super().__init__(vocab_size, model_dim, obj_selection_cfg, pos_generation_cfg) 
        self.distill_encoder = DistillEncoder(model_dim=model_dim,ignore_rgb=pos_generation_cfg["ignore_rgb"])
        
        # same as position transformer
        self.distill_encoder.object_encoder = self.pos_generation_model.object_encoder
        self.distill_encoder.object_mlp =  self.pos_generation_model.mlp
        
    def forward(self, xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask, sentence, sentence_pad_mask, token_type_index, obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index, tgt_mask, start_token, struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs, struct_position_index, struct_token_type_index, struct_pad_mask, 
                origin_xyzs,origin_rgbs,transform_xyzs,transform_rgbs,
                batch_size=None,):
        
        print(xyzs.shape)
        print(origin_xyzs.shape)
        
        if sentence_pad_mask is not None:
            batch_size, _ = sentence_pad_mask.size()
        device = xyzs.device

        max_obj = xyzs.size(0) // batch_size
        max_other_obj = other_xyzs.size(0) // batch_size
        sel_input_dim = batch_size * (max_obj + max_other_obj)
        sel_num_points = xyzs.size(1)

        # get utility   
        utility_inputs = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        utility_embed = self.utility_embeddings(utility_inputs)
        if sentence is not None:
            word_embed = self.word_embeddings(sentence)
            # in args pad 0 means valid
            word_pad_mask = 1 - sentence_pad_mask.reshape(batch_size, -1, 1).float()
            word_embed = word_embed * word_pad_mask

            # sum as utility embedding
            word_sum_embed = torch.sum(word_embed, dim=1, keepdim=True)
            # avg pooling
            utility_embed = word_sum_embed
        
        distill_embed = self.distill_encoder(origin_xyzs,origin_rgbs,transform_xyzs,transform_rgbs)
        utility_embed = utility_embed + distill_embed

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