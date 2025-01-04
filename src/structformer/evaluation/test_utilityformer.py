import torch
import numpy as np
import os
import copy
import tqdm
import argparse
from omegaconf import OmegaConf
import time
from rich import print

from torch.utils.data import DataLoader
from structformer.models.utilityformer import UtilityFomrer
from structformer.models.pose_generation_network import UtilityFormerPoseGenerationWSentence
from structformer.data.tokenizer import Tokenizer
from structformer.data.sequence_dataset import SequenceDataset
import structformer.training.train_utilityformer as prior_model
from structformer.utils.rearrangement import show_pcs
from structformer.evaluation.inference import PointCloudRearrangement
from structformer.utils.rearrangement import evaluate_prior_prediction, generate_square_subsequent_mask
from structformer.training.train_object_selection_network import evaluate
from structformer.utils import load_json_file,dump_json_file
class PriorUtilityInference:

    def __init__(self, model_dir, dirs_cfg, data_split="test"):

        cfg, tokenizer, model, _, _, epoch = prior_model.load_model(model_dir, dirs_cfg)
        #prior_model.save_model(model_dir+"_with_ue",cfg,epoch,model)

        data_cfg = cfg.dataset

        dataset = SequenceDataset(data_cfg.dirs, data_cfg.index_dirs, data_split, tokenizer,
                                                data_cfg.max_num_objects,
                                                data_cfg.max_num_other_objects,
                                                data_cfg.max_num_shape_parameters,
                                                data_cfg.max_num_rearrange_features,
                                                data_cfg.max_num_anchor_features,
                                                data_cfg.num_pts,
                                                data_cfg.use_structure_frame)

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = cfg
        self.dataset = dataset
        self.epoch = epoch


    def limited_batch_inference(self, data, use_utility_vector=False,verbose=False):
        """
        This function makes the assumption that scenes in the batch have the same number of objects that need to be
        rearranged

        :param data:
        :param model:
        :param test_dataset:
        :param tokenizer:
        :param cfg:
        :param num_samples:
        :param verbose:
        :return:
        """
        data_size = len(data)
        batch_size = self.cfg.dataset.batch_size
        if verbose:
            print("data size:", data_size)
            print("batch size:", batch_size)

        num_batches = int(data_size / batch_size)
        if data_size % batch_size != 0:
            num_batches += 1
            
        all_obj_preds = []
        all_struct_preds = []
        for b in range(num_batches):
            if b + 1 == num_batches:
                # last batch
                batch = data[b * batch_size:]
            else:
                batch = data[b * batch_size: (b+1) * batch_size]
            print("do not use sentence:",{use_utility_vector})
            data_tensors = [self.dataset.convert_to_tensors(d, self.tokenizer,use_utility_vector) for d in batch]
            data_tensors = self.dataset.collate_fn(data_tensors)
            predictions = prior_model.infer_once(self.cfg, self.model, data_tensors, self.cfg.device)

            obj_x_preds = torch.cat(predictions["obj_x_outputs"], dim=0)
            obj_y_preds = torch.cat(predictions["obj_y_outputs"], dim=0)
            obj_z_preds = torch.cat(predictions["obj_z_outputs"], dim=0)
            obj_theta_preds = torch.cat(predictions["obj_theta_outputs"], dim=0)
            obj_preds = torch.cat([obj_x_preds, obj_y_preds, obj_z_preds, obj_theta_preds], dim=1)  # batch_size * max num objects, output_dim

            struct_x_preds = torch.cat(predictions["struct_x_inputs"], dim=0)
            struct_y_preds = torch.cat(predictions["struct_y_inputs"], dim=0)
            struct_z_preds = torch.cat(predictions["struct_z_inputs"], dim=0)
            struct_theta_preds = torch.cat(predictions["struct_theta_inputs"], dim=0)
            struct_preds = torch.cat([struct_x_preds, struct_y_preds, struct_z_preds, struct_theta_preds], dim=1) # batch_size, output_dim

            all_obj_preds.append(obj_preds)
            all_struct_preds.append(struct_preds)
        
        obj_preds = torch.cat(all_obj_preds, dim=0)  # data_size * max num objects, output_dim
        struct_preds = torch.cat(all_struct_preds, dim=0)  # data_size, output_dim

        obj_preds = obj_preds.detach().cpu().numpy()
        struct_preds = struct_preds.detach().cpu().numpy()

        obj_preds = obj_preds.reshape(data_size, -1, obj_preds.shape[-1])  # batch_size, max num objects, output_dim

        return struct_preds, obj_preds
        
        


def inference_beam_decoding(model_dir, dirs_cfg, beam_size=3, max_scene_decodes=3,
                            visualize=True, visualize_action_sequence=False, use_utility_vector=False,
                            inference_visualization_dir=None):
    """

    :param model_dir:
    :param beam_size:
    :param max_scene_decodes:
    :param visualize:
    :param visualize_action_sequence:
    :param inference_visualization_dir:
    :param side_view:
    :return:
    """
    if inference_visualization_dir and not os.path.exists(inference_visualization_dir):
        os.makedirs(inference_visualization_dir)

    prior_inference = PriorUtilityInference(model_dir, dirs_cfg)
    test_dataset = prior_inference.dataset
    
    if inference_visualization_dir:
        instruction_path = os.path.join(inference_visualization_dir, "index.json")

    decoded_scene_count = 0
    with tqdm.tqdm(total=len(test_dataset)) as pbar:
        # for idx in np.random.choice(range(len(test_dataset)), len(test_dataset), replace=False):
        for idx in range(len(test_dataset)):

            if decoded_scene_count >= max_scene_decodes:
                break

            filename = test_dataset.get_data_index(idx)
            scene_id = os.path.split(filename)[1][4:-3]

            decoded_scene_count += 1

            ############################################
            # retrieve data
            beam_data = []
            beam_pc_rearrangements = []
            natural_language_instruction = ""
            for b in range(beam_size):
                datum = test_dataset.get_raw_data(idx, inference_mode=True, shuffle_object_index=False)
                
                if not use_utility_vector:
                    natural_language_instruction = test_dataset.tokenizer.convert_structure_params_to_natural_language(datum["sentence"])

                # not necessary, but just to ensure no test leakage
                datum["struct_x_inputs"] = [0]
                datum["struct_y_inputs"] = [0]
                datum["struct_y_inputs"] = [0]
                datum["struct_theta_inputs"] = [[0] * 9]
                for obj_idx in range(len(datum["obj_x_inputs"])):
                    datum["obj_x_inputs"][obj_idx] = 0
                    datum["obj_y_inputs"][obj_idx] = 0
                    datum["obj_z_inputs"][obj_idx] = 0
                    datum["obj_theta_inputs"][obj_idx] = [0] * 9

                beam_data.append(datum)
                beam_pc_rearrangements.append(PointCloudRearrangement(datum))
            
            if inference_visualization_dir:
                for pc_rearrangement in beam_pc_rearrangements:
                    #initial
                    pc_rearrangement.visualize("initial", add_other_objects=True,
                                               add_coordinate_frame=False, side_view=True, add_table=True,server=True,
                                               save_vis=True,show_vis=False,
                                               save_filename=os.path.join(inference_visualization_dir, "{}_original.jpg".format(scene_id)))
                    break
            
            # autoregressive decoding
            num_target_objects = beam_pc_rearrangements[0].num_target_objects
            # first predict structure pose
            beam_goal_struct_pose, target_object_preds = prior_inference.limited_batch_inference(beam_data,use_utility_vector)
            for b in range(beam_size):
                datum = beam_data[b]
                datum["struct_x_inputs"] = [beam_goal_struct_pose[b][0]]
                datum["struct_y_inputs"] = [beam_goal_struct_pose[b][1]]
                datum["struct_z_inputs"] = [beam_goal_struct_pose[b][2]]
                datum["struct_theta_inputs"] = [beam_goal_struct_pose[b][3:]]

            # then iteratively predict pose of each object
            beam_goal_obj_poses = []
            for obj_idx in range(num_target_objects):
                struct_preds, target_object_preds = prior_inference.limited_batch_inference(beam_data,use_utility_vector)
                beam_goal_obj_poses.append(target_object_preds[:, obj_idx])
                for b in range(beam_size):
                    datum = beam_data[b]
                    datum["obj_x_inputs"][obj_idx] = target_object_preds[b][obj_idx][0]
                    datum["obj_y_inputs"][obj_idx] = target_object_preds[b][obj_idx][1]
                    datum["obj_z_inputs"][obj_idx] = target_object_preds[b][obj_idx][2]
                    datum["obj_theta_inputs"][obj_idx] = target_object_preds[b][obj_idx][3:]
            # concat in the object dim
            beam_goal_obj_poses = np.stack(beam_goal_obj_poses, axis=0)
            # swap axis
            beam_goal_obj_poses = np.swapaxes(beam_goal_obj_poses, 1, 0)  # batch size, number of target objects, pose dim
            
            ############################################
            # move pc
            for bi in range(beam_size):

                beam_pc_rearrangements[bi].set_goal_poses(beam_goal_struct_pose[bi], beam_goal_obj_poses[bi])
                beam_pc_rearrangements[bi].rearrange()
            
            # Visiualize
            if inference_visualization_dir:
                
                index_file = load_json_file(instruction_path,data_type="dict")
                if not use_utility_vector:
                    index_file[scene_id] = natural_language_instruction
                dump_json_file(index_file,instruction_path,if_backup=False)
                
                #TODO 把它加到instruction中  #natural_language_instruction
                for pc_rearrangement in beam_pc_rearrangements:
                    pc_rearrangement.visualize("goal", add_other_objects=True,
                                               add_coordinate_frame=False, side_view=True, add_table=True,server=True,
                                               save_vis=True,show_vis=False,
                                               save_filename=os.path.join(inference_visualization_dir, "{}.jpg".format(scene_id)))
                    break
            
            pbar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", default="/home/lmy/workspace/Structformer/data/data_new_objects",help='location of the dataset', type=str)
    parser.add_argument("--main_config", help='config yaml file for the model',
                        default='configs/utilityformer.yaml',
                        type=str)
    parser.add_argument("--model_dir",default="model/24-12-09",help='location for the saved model', type=str) #"experiments/yutang"
    parser.add_argument("--dirs_config", help='config yaml file for directories',
                        default='configs/data/circle_dirs.yaml',
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
    cfg.dataset_base_dir = args.dataset_base_dir
    OmegaConf.resolve(cfg)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))
    
    inference_beam_decoding(args.model_dir, dirs_cfg, **cfg.inference)
