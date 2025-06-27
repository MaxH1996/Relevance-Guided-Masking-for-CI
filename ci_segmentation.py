import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights # Example
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import numpy as np
import os
import math
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import argparse
from typing import Optional, Union, List
# --- LRP/CRP Imports ---
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer
from crp.attribution import CondAttribution # Assuming this is your base CondAttribution
# --- NNCodec Tensor Coding Imports ---
from nncodec.tensor import encode, decode
# --- other compressors
import bz2

CRP_ZENNIT_AVAILABLE = True # Assume available
from collections import OrderedDict
import torch.nn.functional as F
# --- Configuration ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR_VOC = '/Users/adm_becking/PycharmProjects/data/' #/data2/datasets/' #'/Users/adm_becking/PycharmProjects/data/VOCdevkit'   #'/home/hoefler/remote_inference/data/voc_segmentation'
MODEL_SAVE_BASE_VOC = './models_ci/deeplabv3_voc'
PLOT_DIR_VOC = './plots_ci/voc_deeplab'
BS_DIR_VOC = './bitstreams_ci/voc_deeplab'
NUM_VOC_CLASSES = 21
IGNORE_INDEX_VOC = 255
INPUT_SIZE_HW_VOC = (256, 256)

os.makedirs(MODEL_SAVE_BASE_VOC, exist_ok=True)
os.makedirs(PLOT_DIR_VOC, exist_ok=True)
os.makedirs(BS_DIR_VOC, exist_ok=True)

# --- Helper Functions (from your VGG script) ---
def count_zeros(tensor):
    if not isinstance(tensor, torch.Tensor): return 0
    return (tensor == 0).sum().item()

def normalize_relevance(relevance):
    if not isinstance(relevance, torch.Tensor) or relevance.numel() == 0: return relevance
    relevance = relevance.float(); min_val, max_val = torch.min(relevance), torch.max(relevance)
    range_val = max_val - min_val; return (relevance - min_val) / (range_val + 1e-10)

# --- Data Transforms and Loading (VOC Specific) ---
class TargetTransformVOC:
    def __call__(self, target_mask_pil):
        target_mask_np = np.array(target_mask_pil, dtype=np.int64)
        return torch.from_numpy(target_mask_np)

def get_voc_transforms(input_size_hw): # For images
    return {
        'train': transforms.Compose([
            transforms.Resize(input_size_hw, interpolation=transforms.InterpolationMode.BILINEAR),
            #transforms.RandomResizedCrop(input_size_hw, scale=(0.8, 1.2), ratio=(0.75, 1.33)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size_hw, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

def get_voc_target_transforms(input_size_hw): # For masks
    return {
        'train': transforms.Compose([
            transforms.Resize(input_size_hw, interpolation=transforms.InterpolationMode.NEAREST),
            TargetTransformVOC()
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size_hw, interpolation=transforms.InterpolationMode.NEAREST),
            TargetTransformVOC()
        ])
    }

def load_voc_data_for_ci(data_dir, batch_size, input_size_hw, num_workers=0):
    voc_transforms = get_voc_transforms(input_size_hw)
    voc_target_transforms = get_voc_target_transforms(input_size_hw)
    image_datasets = {
        'train': datasets.VOCSegmentation(root=data_dir, year='2012', image_set='train',
                                          download=False, transform=voc_transforms['train'],
                                          target_transform=voc_target_transforms['train']),
        'val': datasets.VOCSegmentation(root=data_dir, year='2012', image_set='val',
                                        download=False, transform=voc_transforms['val'],
                                        target_transform=voc_target_transforms['val'])
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                       shuffle=(x == 'train'), num_workers=num_workers, pin_memory=True)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"VOC Data: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}")
    return dataloaders, dataset_sizes

# --- Corrected CondAttribution for Segmentation (Handles full GT mask) ---
class CondAttributionSegWithFullGT(CondAttribution):
    def __init__(self, model: nn.Module, device_override=None):
        super().__init__(model)
        # ... (device setup as before) ...
        self.rel_init_strategy = "zplus_on_gt_pixels"


    def relevance_init(self,
                       prediction: torch.Tensor,
                       target_list: Optional[Union[torch.Tensor, List[torch.Tensor]]], # MODIFIED TYPE HINT
                       init_rel: Optional[torch.Tensor]
                      ) -> torch.Tensor:
        if not hasattr(self, 'device') or self.device is None: self.device = prediction.device
        if init_rel is not None: return init_rel

        if target_list is None:
            print("Warning: target_list (ground truth) is None in relevance_init. Applying Z+ to all predictions.")
            clamped_prediction = prediction.clamp(min=0)
            sum_val = clamped_prediction.sum(dim=(1,2,3), keepdim=True) + 1e-9
            return clamped_prediction / sum_val

        # --- FIX FOR 'list' object has no attribute 'to' ---
        if isinstance(target_list, list):
            # If it's a list of tensors, stack them to form a batch.
            # This assumes all tensors in the list have the same H, W dimensions.
            try:
                y_true_batched = torch.stack(target_list, dim=0)
            except RuntimeError as e:
                print(f"Error stacking target_list (list of tensors): {e}")
                print("Individual tensor shapes in list:")
                for i, t in enumerate(target_list):
                    print(f"  target_list[{i}] shape: {t.shape if isinstance(t, torch.Tensor) else type(t)}")
                # Fallback or re-raise
                clamped_prediction = prediction.clamp(min=0)
                sum_val = clamped_prediction.sum(dim=(1,2,3), keepdim=True) + 1e-9
                return clamped_prediction / sum_val # Fallback if stacking fails
        elif isinstance(target_list, torch.Tensor):
            y_true_batched = target_list
        else:
            raise TypeError(f"target_list expected to be a torch.Tensor or List[torch.Tensor], got {type(target_list)}")
        # --- END FIX ---

        y_true = y_true_batched.to(self.device) # Now y_true is guaranteed to be a tensor [B, H, W]
        
        r = torch.zeros_like(prediction)    # [B, C, H, W]
        valid_mask = (y_true != IGNORE_INDEX_VOC) # [B, H, W]

        if self.rel_init_strategy == "zplus_on_gt_pixels":
            for i in range(prediction.shape[0]): # Iterate over batch
                sample_prediction = prediction[i] # [C, H, W]
                sample_y_true = y_true[i]         # [H, W]
                sample_valid_mask = valid_mask[i] # [H, W]

                y_clamped_for_gather = sample_y_true.clone()
                y_clamped_for_gather[~sample_valid_mask] = 0 

                class_indices_for_scatter = y_clamped_for_gather.unsqueeze(0)
                true_class_logits_at_pixels = torch.gather(sample_prediction, 0, class_indices_for_scatter).squeeze(0)
                
                relevance_values = true_class_logits_at_pixels.clamp(min=0) 
                relevance_values_masked = relevance_values * sample_valid_mask.float()

                r[i].scatter_(0, class_indices_for_scatter, relevance_values_masked.unsqueeze(0))
        else:
            raise ValueError(f"Unsupported rel_init_strategy: {self.rel_init_strategy}")

        sum_r_per_sample = r.sum(dim=(1, 2, 3), keepdim=True) + 1e-9
        init_rel_final = r / sum_r_per_sample
        return init_rel_final

# --- SegModelWrapper (for LRP, as CondAttribution expects single tensor output) ---
class SegModelWrapper(nn.Module):
    def __init__(self, seg_model: nn.Module, head_name="out"):
        super().__init__(); self.model = seg_model; self.head_name = head_name
    def forward(self, x):
        output_dict = self.model(x)
        if self.head_name not in output_dict: raise KeyError(f"Head missing. Avail: {list(output_dict.keys())}")
        return output_dict[self.head_name]

class DeepLabCI_VOC(nn.Module):
    def __init__(self, num_classes, pretrained_checkpoint_path, split_layer_name_str: str,
                 model_arch='deeplabv3_resnet50', aux_loss_at_load=True, input_size_hw=(256,256)): # Added input_size_hw
        super().__init__()
        self.base_model_name = f"{model_arch}_voc_ci_split"
        self.split_layer_name_str = split_layer_name_str
        self.model_arch = model_arch
        self.input_size_hw_at_init = input_size_hw # Store for upsampling in rest()
        self.num_classes = num_classes # Store for clarity

        # 1. Load base model
        if model_arch == 'deeplabv3_resnet50':
            self.base_deeplab_model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=self.num_classes, aux_loss=aux_loss_at_load)
        elif model_arch == 'deeplabv3_resnet101':
            self.base_deeplab_model = models.segmentation.deeplabv3_resnet101(weights=None, num_classes=self.num_classes, aux_loss=aux_loss_at_load)
        else:
            raise ValueError(f"Unsupported model_arch: {model_arch}")
        try:
            state_dict = torch.load(pretrained_checkpoint_path, map_location='cpu')
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.base_deeplab_model.load_state_dict(state_dict)
            print(f"Loaded fine-tuned {model_arch} from {pretrained_checkpoint_path}")
        except Exception as e:
            print(f"Error loading fine-tuned model checkpoint: {e}"); raise

        # 2. Perform the split to create self.client_part and self.server_part
        self._construct_client_server_parts()

        # This is the target for LRP map generation on the base_deeplab_model
        self.lrp_target_layer_in_base = self.split_layer_name_str

    def _get_module_by_path(self, model_to_search, path_str: str):
        parts = path_str.split('.')
        current_module = model_to_search
        for p_name in parts:
            if not hasattr(current_module, p_name): return None
            current_module = getattr(current_module, p_name)
        return current_module

    def _add_modules_recursively(self, base_module, path_prefix, target_split_path,
                                 current_container, found_split_flag, collecting_for_server):
        """
        Recursively adds children of base_module to current_container.
        Switches to collecting_for_server=True after target_split_path is processed.
        Returns True if the split point was handled within this call or its children.
        """
        split_handled_in_this_branch = False
        for name, child_module in base_module.named_children():
            full_child_path = f"{path_prefix}.{name}" if path_prefix else name

            if found_split_flag[0]: # Split point already passed globally
                if collecting_for_server:
                    current_container.add_module(name, copy.deepcopy(child_module)) # Deepcopy for server
                continue # Skip adding to client if already past split

            # If this child IS the split point
            if full_child_path == target_split_path:
                if not collecting_for_server: # Add to client
                    current_container.add_module(name, copy.deepcopy(child_module))
                found_split_flag[0] = True
                split_handled_in_this_branch = True
                # Subsequent siblings will be handled by the found_split_flag[0] check above
            
            # If this child CONTAINS the split point (split is deeper)
            elif target_split_path.startswith(full_child_path + "."):
                if not collecting_for_server:
                    new_sub_container_client = nn.Sequential() # Create a new sequential for this part of client
                    current_container.add_module(name, new_sub_container_client)
                    # Recurse into this child to populate its client and potentially start its server part
                    self._add_modules_recursively(child_module, full_child_path, target_split_path,
                                                  new_sub_container_client, found_split_flag, collecting_for_server)
                    # If recursion found the split, this branch is done for client adding
                    if found_split_flag[0]:
                         split_handled_in_this_branch = True
                # Server part will be built in a separate pass or by a different logic
                # This recursive build for client is the priority here.
            
            # If this child is entirely BEFORE the split point path
            elif not target_split_path.startswith(full_child_path): # and full_child_path < target_split_path (approx)
                if not collecting_for_server:
                    current_container.add_module(name, copy.deepcopy(child_module))
            
            # If this child is entirely AFTER where the split would have been (should be caught by found_split_flag)
            # (This case is mostly for server construction pass)
            elif collecting_for_server:
                 current_container.add_module(name, copy.deepcopy(child_module))


        return split_handled_in_this_branch


    def _construct_client_server_parts(self):
        self.client_part = nn.Sequential()
        self.server_part = nn.Sequential() # Will hold the main execution path for server

        # Create ordered dicts for client and server parts of backbone and classifier
        client_backbone_layers = OrderedDict()
        server_backbone_layers = OrderedDict()
        client_classifier_layers = OrderedDict()
        server_classifier_layers = OrderedDict()

        found_split = False

        # 1. Process Backbone
        for name, child_module in self.base_deeplab_model.backbone.named_children():
            full_child_name = f"backbone.{name}"
            if found_split:
                server_backbone_layers[name] = child_module # Keep original reference for server part
            else:
                client_backbone_layers[name] = child_module # Keep original reference for client part
                if full_child_name == self.split_layer_name_str:
                    found_split = True
        
        if client_backbone_layers:
            self.client_part.add_module("backbone", nn.Sequential(client_backbone_layers))
        if server_backbone_layers:
            self.server_part.add_module("backbone", nn.Sequential(server_backbone_layers))

        # 2. Process Classifier
        if hasattr(self.base_deeplab_model, 'classifier'):
            if not found_split: # Split must be in classifier (or is 'classifier' itself)
                if self.split_layer_name_str == "classifier": # Split after entire classifier module
                    self.client_part.add_module("classifier", self.base_deeplab_model.classifier)
                    found_split = True
                else: # Split is within classifier
                    for name, child_module in self.base_deeplab_model.classifier.named_children():
                        full_child_name = f"classifier.{name}"
                        if found_split:
                            server_classifier_layers[name] = child_module
                        else:
                            client_classifier_layers[name] = child_module
                            if full_child_name == self.split_layer_name_str:
                                found_split = True
                    if client_classifier_layers:
                        self.client_part.add_module("classifier", nn.Sequential(client_classifier_layers))
                    if server_classifier_layers:
                        self.server_part.add_module("classifier", nn.Sequential(server_classifier_layers))
            elif hasattr(self.server_part, "backbone") or not client_backbone_layers : # If split was in backbone OR backbone client part is empty
                # Add full classifier to server part if it wasn't already processed
                if not server_classifier_layers: # Check if not already populated
                     self.server_part.add_module("classifier", self.base_deeplab_model.classifier)


        # 3. Handle Aux Classifier (always on server for this CI model's `rest` logic)
        if hasattr(self.base_deeplab_model, 'aux_classifier') and self.base_deeplab_model.aux_classifier is not None:
            self.server_part_aux_classifier = self.base_deeplab_model.aux_classifier
        else:
            self.server_part_aux_classifier = None

        if not found_split:
            raise ValueError(f"Split layer name '{self.split_layer_name_str}' not found or not handled by splitting logic.")
        
        print(f"Constructed client/server parts. Split at: {self.split_layer_name_str}")
        # print("Client Part:\n", self.client_part)
        # print("Server Part:\n", self.server_part)


    def intermediate_for_shape_and_lrp(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        Precisely gets the output of self.split_layer_name_str from base_deeplab_model using a hook.
        This is used for get_intermediate_activation_shape_voc and LRP map generation.
        """
        target_module = self._get_module_by_path(self.base_deeplab_model, self.split_layer_name_str)
        if target_module is None:
            # Try to find it within backbone or classifier explicitly if path is short
            if self.split_layer_name_str.startswith("backbone."):
                 target_module = self._get_module_by_path(self.base_deeplab_model.backbone, self.split_layer_name_str.split('.',1)[1])
            elif self.split_layer_name_str.startswith("classifier."):
                 target_module = self._get_module_by_path(self.base_deeplab_model.classifier, self.split_layer_name_str.split('.',1)[1])
            
            if target_module is None: # Still not found
                print("Available modules in base_deeplab_model:")
                for name, _ in self.base_deeplab_model.named_modules(): print(name)
                raise ValueError(f"Split layer '{self.split_layer_name_str}' not found in base model for hook-based intermediate.")

        activation_capture = {}
        def hook_fn(module, input, output):
            activation_capture['out'] = output.detach().clone()

        handle = target_module.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = self.base_deeplab_model(x_image) # Run full model to trigger hook

        handle.remove()
        if 'out' not in activation_capture:
            raise RuntimeError(f"Hook failed to capture output for {self.split_layer_name_str}. Check module path and execution.")
        
        return activation_capture['out']

    def intermediate(self, x_image: torch.Tensor) -> torch.Tensor: # For actual split inference
        """Executes the constructed client_part."""
        # This relies on self.client_part being correctly constructed by _construct_client_server_parts
        current_features = x_image
        if hasattr(self.client_part, 'backbone'):
            current_features = self.client_part.backbone(current_features)
        
        if hasattr(self.client_part, 'classifier'):
            current_features = self.client_part.classifier(current_features)
            
        return current_features

    def rest(self, x_intermediate: torch.Tensor) -> torch.Tensor: # For actual split inference
        """Executes the constructed server_part and handles upsampling."""
        current_features = x_intermediate
        main_output_low_res = None
        
        # If server_part has a 'backbone'
        if hasattr(self.server_part, 'backbone'):
            current_features = self.server_part.backbone(current_features)
        
        # If server_part has a 'classifier'
        if hasattr(self.server_part, 'classifier'):
            main_output_low_res = self.server_part.classifier(current_features)
        else: # Client did everything up to low-res logits, or server_part has no classifier component
            main_output_low_res = current_features 

        # Final Upsampling for main output
        if main_output_low_res.shape[-2:] != self.input_size_hw_at_init:
            logits_upsampled = F.interpolate(
                main_output_low_res, size=self.input_size_hw_at_init, mode='bilinear', align_corners=False
            )
        else:
            logits_upsampled = main_output_low_res
            
        # Note: Aux classifier output from self.server_part_aux_classifier is not used here for simplicity.
        # If needed for server-side fine-tuning loss, `rest` would return a dict.
        return logits_upsampled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For baseline evaluation, use the original model's forward pass (which includes upsampling)
        return self.base_deeplab_model(x)['out']

    def get_client_params(self):
        return list(self.client_part.parameters())

    def get_server_params(self):
        params = list(self.server_part.parameters())
        if self.server_part_aux_classifier is not None: # Add aux params if they exist on server
            params.extend(list(self.server_part_aux_classifier.parameters()))
        return params

# --- get_intermediate_activation_shape_voc (Mostly Unchanged) ---
def get_intermediate_activation_shape_voc(model_ci: DeepLabCI_VOC, sample_input_loader, device, current_config_dict):
    model_ci.eval()
    try:
        sample_images, _ = next(iter(sample_input_loader))
        # Use the hook-based method for precise shape determination
        sample_intermediate_act = model_ci.intermediate_for_shape_and_lrp(sample_images[0:1].to(device))
        shape = sample_intermediate_act.shape[1:] 
        del sample_images, sample_intermediate_act; torch.cuda.empty_cache()
        return shape
    except Exception as e:
        print(f"Error get_intermediate_activation_shape_voc (using hook method): {e}")
        # ... (Your existing fallback logic, but hopefully not needed as much) ...
        split_name = model_ci.split_layer_name_str
        H, W = model_ci.input_size_hw_at_init # Use from model instance
        s4_h,s4_w=H//4,W//4; s8_h,s8_w=H//8,W//8; s16_h,s16_w=H//16,W//16

        if split_name == "backbone.conv1": return torch.Size([64, H//2, W//2])
        elif split_name == "backbone.bn1": return torch.Size([64, H//2, W//2])
        elif split_name == "backbone.relu": return torch.Size([64, H//2, W//2])
        elif split_name == "backbone.maxpool": return torch.Size([64, s4_h, s4_w])
        elif split_name == "backbone.layer1": return torch.Size([256, s4_h, s4_w])
        elif split_name == "backbone.layer2": return torch.Size([512, s8_h, s8_w])
        elif split_name == "backbone.layer3": return torch.Size([1024, s16_h, s16_w])
        elif split_name == "backbone.layer4": return torch.Size([2048, s16_h, s16_w])
        elif split_name.endswith(".relu"): # Heuristic for relu outputs of blocks
            if "layer1" in split_name: return torch.Size([256, s4_h, s4_w])
            if "layer2" in split_name: return torch.Size([512, s8_h, s8_w])
            if "layer3" in split_name: return torch.Size([1024, s16_h, s16_w])
            if "layer4" in split_name: return torch.Size([2048, s16_h, s16_w])
        elif split_name == "classifier.0.project.0": return torch.Size([256, s16_h, s16_w])
        elif split_name == "classifier.3": return torch.Size([256, s16_h, s16_w])
        print(f"No specific fallback for DeepLab split: {split_name}.")
        raise e

# --- CondAttribution for Segmentation (Revised for more flexible init) ---
class CondAttributionSegFlexible(CondAttribution):
    def __init__(self, model: nn.Module, device_override=None):
        super().__init__(model)
        # ... (device setup) ...
        if device_override is not None: self.device = device_override
        elif hasattr(model, 'device'): self.device = model.device
        else:
             try: self.device = next(model.parameters()).device
             except StopIteration: self.device = torch.device("cpu")

    def to(self, device): # ... (same) ...
        super().to(device); self.device = device; return self

    def relevance_init(self,
                       prediction: torch.Tensor,    # Logits [B, C, H, W]
                       conditions: List[dict],      # List of dicts, one per sample
                                                    # Each dict: {'y': target_class_id (scalar), 
                                                    #             'target_mask': binary_mask_for_target_class [H,W]}
                       init_rel_passed: Optional[torch.Tensor]
                      ) -> torch.Tensor:
        # prediction is [B, NumClasses, H, W]
        # conditions is a list of B dictionaries.
        # We expect each dict to have 'y' (the target class ID for this explanation)
        # and 'target_mask' (a [H,W] binary mask for where to initialize relevance for class 'y')

        if not hasattr(self, 'device') or self.device is None: self.device = prediction.device
        if init_rel_passed is not None: return init_rel_passed

        if not conditions or not isinstance(conditions, list) or not isinstance(conditions[0], dict):
            print("Warning: `conditions` not in expected format (List[dict]). Applying Z+ to all predictions.")
            clamped_prediction = prediction.clamp(min=0)
            sum_val = clamped_prediction.sum(dim=(1,2,3), keepdim=True) + 1e-9
            return clamped_prediction / sum_val

        r = torch.zeros_like(prediction).to(self.device)

        for i in range(prediction.shape[0]): # Iterate over batch
            if i >= len(conditions): # Should not happen if conditions match batch size
                print(f"Warning: Mismatch between prediction batch size {prediction.shape[0]} and len(conditions) {len(conditions)}")
                continue

            condition = conditions[i]
            target_class_id = condition.get('y') # Scalar class ID
            # Binary mask [H,W] for this target_class_id for this sample
            binary_spatial_mask_for_target = condition.get('target_mask') 

            if target_class_id is None or binary_spatial_mask_for_target is None:
                print(f"Warning: Sample {i} missing 'y' or 'target_mask' in conditions. Skipping relevance init for it.")
                continue
            
            binary_spatial_mask_for_target = binary_spatial_mask_for_target.to(self.device).float() # Ensure float

            if not (0 <= target_class_id < prediction.shape[1]):
                print(f"Warning: Sample {i} has invalid target_class_id {target_class_id}. Skipping.")
                continue
            if binary_spatial_mask_for_target.shape != prediction.shape[-2:]:
                print(f"Warning: Sample {i} target_mask shape {binary_spatial_mask_for_target.shape} "
                      f"mismatches prediction spatial shape {prediction.shape[-2:]}. Skipping.")
                continue

            # Strategy: Z+ on specified class channel, only at pixels in binary_spatial_mask_for_target
            logits_for_target_class_channel = prediction[i, target_class_id, :, :] # [H,W]
            relevance_values = logits_for_target_class_channel.clamp(min=0) # Z+
            
            # Apply the spatial mask
            relevance_values_masked = relevance_values * binary_spatial_mask_for_target
            
            r[i, target_class_id, :, :] = relevance_values_masked
        
        # Normalize per sample
        sum_r_per_sample = r.sum(dim=(1, 2, 3), keepdim=True)
        # Handle cases where a sample might have zero initial relevance (e.g. no target pixels)
        # To avoid NaN/Inf, if sum is zero for a sample, its relevance remains zero.
        init_rel_final = torch.where(sum_r_per_sample.abs() > 1e-9, r / (sum_r_per_sample + 1e-9), torch.zeros_like(r))
        
        # Debug print
        # print(f"Debug relevance_init: init_rel_final sum per sample = {init_rel_final.sum(dim=(1,2,3))}")
        return init_rel_final

# --- Mask Creation & Calibration (Unchanged from your working versions) ---
# create_thresholded_mask_from_map, create_random_binary_mask,
# compute_average_abs_activations, calibrate_value_mask_for_activation_sparsity
# (Paste them here from previous script if not already present)
def create_thresholded_mask_from_map(input_map, target_mask_sparsity_percent, return_binary_indicator=False):
    if not isinstance(input_map, torch.Tensor): raise TypeError("Input map must be a torch.Tensor")
    if not 0 <= target_mask_sparsity_percent < 100: raise ValueError("Target mask sparsity must be in [0, 100)")
    percent_to_keep = 100.0 - target_mask_sparsity_percent
    if percent_to_keep >= 100.0: return torch.ones_like(input_map) if return_binary_indicator else input_map.clone()
    if percent_to_keep <= 0.0: return torch.zeros_like(input_map)
    flat_map_abs = input_map.abs().flatten()
    num_elements = flat_map_abs.numel()
    if num_elements == 0: return torch.ones_like(input_map) if return_binary_indicator else input_map.clone()
    k = int(num_elements * (percent_to_keep / 100.0)); k = max(1, k); k = min(k, num_elements)
    try:
        if k == num_elements: threshold = flat_map_abs.min() - 1e-9 if num_elements > 0 else 0
        else: threshold = torch.topk(flat_map_abs, k).values[-1] - 1e-9
    except Exception as e:
        print(f"Warning: Error getting threshold for k={k}, numel={num_elements}. Defaulting to keep all. Error: {e}")
        return torch.ones_like(input_map) if return_binary_indicator else input_map.clone()
    indicator_mask = (input_map.abs() > threshold).float()
    return indicator_mask if return_binary_indicator else input_map * indicator_mask

def create_random_binary_mask(intermediate_shape, target_mask_sparsity_percent, device):
    if not 0 <= target_mask_sparsity_percent < 100: raise ValueError("Target mask sparsity must be in [0, 100)")
    num_elements = np.prod(intermediate_shape)
    num_zeros = int(num_elements * (target_mask_sparsity_percent / 100.0)); num_ones = num_elements - num_zeros
    mask_flat = torch.cat([torch.ones(num_ones, device=device), torch.zeros(num_zeros, device=device)]) # create on device
    mask_flat = mask_flat[torch.randperm(num_elements, device=device)]
    return mask_flat.reshape(intermediate_shape) # Shape already on device

def compute_average_abs_activations(model_ci, loader, device):
    print(f"Computing average absolute intermediate activations...")
    model_ci.eval(); sum_abs_activations = None; num_samples = 0; batches_processed = 0
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Avg Abs Activations"):
            inputs = inputs.to(device); intermediate_acts = model_ci.intermediate(inputs); abs_acts = torch.abs(intermediate_acts)
            if sum_abs_activations is None: sum_abs_activations = torch.zeros_like(abs_acts[0], device='cpu')
            sum_abs_activations += abs_acts.mean(dim=0).cpu()
            num_samples += inputs.size(0); batches_processed += 1
    if batches_processed == 0 or sum_abs_activations is None: raise RuntimeError("No batches for avg abs activations.")
    avg_abs_activation_map = sum_abs_activations / batches_processed
    print(f"Finished computing average absolute activations over {num_samples} samples.")
    return avg_abs_activation_map

def calibrate_value_mask_for_activation_sparsity(
    model_ci, calibration_loader, base_map_for_masking, target_activation_sparsity, device,
    is_map_binary_creation_logic=False, initial_mask_sparsity_guess=None,
    tolerance=1.0, max_iterations=15, search_step=5.0):
    print(f"\n--- Starting Calibration for Mask ---"); print(f"Target Activation Sparsity: {target_activation_sparsity:.2f}% (+/- {tolerance:.2f}%)")
    print(f"Mask creation: {'Binary from base map' if is_map_binary_creation_logic else 'Weighted Threshold from base map'}")
    max_iterations = int(max_iterations)
    if max_iterations <= 0: print(f"Warning: max_iterations ({max_iterations}) non-positive. Setting to 1."); max_iterations = 1
    if initial_mask_sparsity_guess is None: initial_mask_sparsity_guess = max(0.0, target_activation_sparsity - 20.0)
    current_mask_sparsity_param = np.clip(initial_mask_sparsity_guess, 0.0, 99.9)
    best_mask_sparsity_param = current_mask_sparsity_param
    best_avg_activation_sparsity = -1.0; min_diff = float('inf'); base_map_for_masking_cpu = base_map_for_masking.cpu(); i = 0
    for i in range(max_iterations):
        print(f"\nCalibration Iteration {i+1}/{max_iterations}"); current_mask_sparsity_param = np.clip(current_mask_sparsity_param, 0.0, 99.9)
        print(f" Trying Mask Sparsity Param for base map: {current_mask_sparsity_param:.2f}%")
        trial_mask = create_thresholded_mask_from_map(base_map_for_masking_cpu, current_mask_sparsity_param, return_binary_indicator=is_map_binary_creation_logic)
        trial_mask_dev = trial_mask.to(device); batch_activation_sparsities = []
        model_ci.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(calibration_loader, desc=" Calib Measure", leave=False):
                inputs = inputs.to(device); act_intermediate = model_ci.intermediate(inputs)
                masked_intermediate = act_intermediate * trial_mask_dev.unsqueeze(0); num_elements = masked_intermediate.numel()
                if num_elements > 0: batch_activation_sparsities.append(100.0 * count_zeros(masked_intermediate) / num_elements)
        if not batch_activation_sparsities: print(" Warning: No calib data processed."); current_mask_sparsity_param += search_step; continue
        measured_avg_activation_sparsity = np.mean(batch_activation_sparsities); diff = abs(measured_avg_activation_sparsity - target_activation_sparsity)
        print(f"  Measured Avg Activation Sparsity: {measured_avg_activation_sparsity:.2f}% (Diff: {diff:.2f}%)")
        if diff < min_diff: min_diff = diff; best_mask_sparsity_param = current_mask_sparsity_param; best_avg_activation_sparsity = measured_avg_activation_sparsity
        if diff <= tolerance: print(f"\nCalibration converged!"); break
        adj_factor = search_step * (diff / (tolerance*2 + diff)) if (tolerance*2 + diff) > 1e-6 else search_step
        current_mask_sparsity_param += -adj_factor if measured_avg_activation_sparsity > target_activation_sparsity else adj_factor
        print(f"  Adjusting: New mask param guess: {current_mask_sparsity_param:.2f}%")
    if i == max_iterations - 1 and diff > tolerance: print(f"\nCalibration finished (max iterations: {max_iterations}). Closest found.")
    print(f" Best Mask Sparsity Parameter for base map: {best_mask_sparsity_param:.2f}%"); print(f" Achieved Avg Activation Sparsity (calib set): {best_avg_activation_sparsity:.2f}%")
    return best_mask_sparsity_param

# --- LRP Computation (Corrected, from previous response) ---
def compute_average_relevance_voc(
    full_model_for_lrp, loader, lrp_target_layer_in_full_model: str, 
    combination_strategy="sum", # "sum", "max"
    device=DEVICE
):
    if not CRP_ZENNIT_AVAILABLE: print("CRP/Zennit not available."); return None
    print(f"Computing CLASS-WISE avg LRP for DeepLabV3. Target layer: {lrp_target_layer_in_full_model}")
    
    wrapped_lrp_model = SegModelWrapper(full_model_for_lrp, head_name="out").to(device).eval()
    composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    attribution_explainer = CondAttribution(wrapped_lrp_model) # Use new one

    sum_combined_relevance_cpu = None
    total_images_processed = 0 # Count original images

    for batch_inputs, batch_targets_true_masks in tqdm(loader, desc="Avg LRP (VOC Class-wise)"):
        # Process one image from the batch at a time for LRP calls
        for img_idx_in_batch in range(batch_inputs.shape[0]):
            single_image_gpu = batch_inputs[img_idx_in_batch:img_idx_in_batch+1].to(device) # [1,C,H,W]
            gt_mask_for_image_gpu = batch_targets_true_masks[img_idx_in_batch].to(device)   # [H,W]
            
            if not single_image_gpu.requires_grad:
                single_image_gpu.requires_grad_(True)

            unique_classes = torch.unique(gt_mask_for_image_gpu)
            relevance_maps_for_this_image_classes = [] # Store [1, C_feat, Hf, Wf] maps

            for class_id_tensor in unique_classes:
                class_id = class_id_tensor.item()
                if class_id == 0 or class_id == IGNORE_INDEX_VOC:  # Skip background/ignore
                    continue
                
                # Create binary mask for THIS class_id from the GT mask
                binary_mask_for_class_gt = (gt_mask_for_image_gpu == class_id)  # [H,W] boolean
                if not binary_mask_for_class_gt.any():  # Should not happen if class_id is from unique()
                    continue
                
                # print(f"Original binary mask shape: {binary_mask_for_class_gt.shape}")
                
                # Transform to shape [21, 256, 256] with class dimension
                # Create zeros tensor with shape [num_classes, H, W]
                num_classes = 21  # VOC has 21 classes (including background)
                H, W = binary_mask_for_class_gt.shape
                
                # Initialize tensor of zeros
                binary_mask_one_hot = torch.zeros(num_classes, H, W, 
                                                dtype=binary_mask_for_class_gt.dtype, 
                                                device=binary_mask_for_class_gt.device)
                
                # Set the class_id channel to the binary mask
                binary_mask_one_hot[class_id] = binary_mask_for_class_gt
                
                # print(f"Transformed mask shape: {binary_mask_one_hot.shape}")
                
                
                # Use this for LRP conditions
                lrp_conditions_single_class = [{'y': binary_mask_one_hot}]
                zennit_lrp_target_name = f"model.{lrp_target_layer_in_full_model}"

                try:
                    attr_results = attribution_explainer(
                        single_image_gpu, conditions=lrp_conditions_single_class,
                        composite=composite, record_layer=[zennit_lrp_target_name]
                    )
                    if zennit_lrp_target_name in attr_results.relevances:
                        relevance_for_class = attr_results.relevances[zennit_lrp_target_name] # [1,Cf,Hf,Wf]
                        relevance_maps_for_this_image_classes.append(relevance_for_class)
                    else:
                        print(f" LRP fail for class {class_id}, layer {zennit_lrp_target_name}. Avail: {list(attr_results.relevances.keys())}")

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e): print(f"OOM for LRP class {class_id}"); torch.cuda.empty_cache(); continue
                    else: print(f"RuntimeError LRP class {class_id}: {e}"); import traceback; traceback.print_exc(); continue
                except Exception as e: print(f"Generic Error LRP class {class_id}: {e}"); import traceback; traceback.print_exc(); continue
                finally:
                    if 'attr_results' in locals(): del attr_results # Cleanup
                    if torch.cuda.is_available(): torch.cuda.empty_cache() # Frequent cache clearing

            # Combine relevance maps for the current image
            if relevance_maps_for_this_image_classes:
                # Stack along a new dimension (dim 0), maps are [1, Cf, Hf, Wf]
                # So stacked becomes [NumClassesInImage, 1, Cf, Hf, Wf]
                stacked_relevances_gpu = torch.stack(relevance_maps_for_this_image_classes, dim=0)
                
                combined_relevance_for_image_gpu = None
                if combination_strategy == "sum":
                    combined_relevance_for_image_gpu = torch.sum(stacked_relevances_gpu, dim=0) # Sum over classes, result [1,Cf,Hf,Wf]
                elif combination_strategy == "max":
                    combined_relevance_for_image_gpu = torch.max(stacked_relevances_gpu, dim=0).values # Max over classes
                else: # Default to sum
                    combined_relevance_for_image_gpu = torch.sum(stacked_relevances_gpu, dim=0)
                
                # Accumulate on CPU
                combined_relevance_for_image_cpu = combined_relevance_for_image_gpu.detach().cpu()
                if sum_combined_relevance_cpu is None:
                    sum_combined_relevance_cpu = combined_relevance_for_image_cpu
                else:
                    sum_combined_relevance_cpu += combined_relevance_for_image_cpu
                
                del combined_relevance_for_image_gpu, stacked_relevances_gpu # Cleanup GPU
            
            # Clean up single image data from GPU
            del single_image_gpu, gt_mask_for_image_gpu, relevance_maps_for_this_image_classes
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            total_images_processed +=1 # Increment for each image processed

        # Clean up original batch data (already on CPU if loader uses CPU workers, or move manually)
        del batch_inputs, batch_targets_true_masks
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if total_images_processed == 0 or sum_combined_relevance_cpu is None:
        print("Error: No images successfully processed for LRP.")
        return None
    
    avg_relevance_map_cpu = sum_combined_relevance_cpu / total_images_processed # Avg over all images
    avg_relevance_normalized = normalize_relevance(avg_relevance_map_cpu.squeeze(0)) # Squeeze batch dim (which was 1)
    
    print(f"Finished CLASS-WISE LRP avg relevance over {total_images_processed} images.")
    torch.save(avg_relevance_normalized, "avg_relevance_map_classwise.pt")
    return avg_relevance_normalized


# --- Server Fine-tuning (VOC/DeepLab Specific) ---
def finetune_server_voc(model_ci, finetune_loader, static_mask_tensor, device, lr=1e-5, epochs=10):
    # ... (same as previous response, ensure it uses IGNORE_INDEX_VOC) ...
    print("\n--- Starting Server Fine-tuning (VOC DeepLab) ---"); model_ci.to(device)
    static_mask_dev = static_mask_tensor.to(device)
    for param in model_ci.get_client_params(): param.requires_grad = False
    server_params_to_train = []
    for param in model_ci.get_server_params(): param.requires_grad = True; server_params_to_train.append(param)
    if not server_params_to_train: raise RuntimeError("No trainable server parameters!")
    print(f"Trainable server params: {sum(p.numel() for p in server_params_to_train)}")
    if epochs <= 0: print("Skipping fine-tuning."); return model_ci
    optimizer = optim.Adam(server_params_to_train, lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_VOC)
    for epoch in range(epochs):
        model_ci.train(); running_loss = 0.0
        progress_bar = tqdm(finetune_loader, desc=f"Finetune Ep {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device); optimizer.zero_grad()
            with torch.no_grad(): act_intermediate = model_ci.intermediate(inputs)
            masked_act = act_intermediate * static_mask_dev.unsqueeze(0)
            # print(masked_act.shape)
            outputs_logits = model_ci.rest(masked_act)
            loss = criterion(outputs_logits, targets); loss.backward(); optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_epoch_loss = running_loss / len(finetune_loader.dataset) if len(finetune_loader.dataset) > 0 else 0.0
        print(f"Finetune Server Ep {epoch+1} Loss: {avg_epoch_loss:.4f}")
    print("--- Server Fine-tuning Finished ---"); model_ci.eval(); return model_ci

# --- Plotting (from your VGG script, can be reused) ---
def plot_maps(relevance_map, title="Map", num_channels=6, plot_dir=PLOT_DIR_VOC): # Default to PLOT_DIR_VOC
    # ... (same as previous response, ensure it uses plot_dir) ...
    if not isinstance(relevance_map, torch.Tensor): print(f"Plot err: Expected Tensor for '{title}'"); return
    if relevance_map.numel()==0: print(f"Plot err: map empty for '{title}'"); return
    if relevance_map.dim()==4 and relevance_map.shape[0]==1: relevance_map = relevance_map.squeeze(0)
    if relevance_map.dim()!=3: print(f"Plot err: Expected 3D (C,H,W), got {relevance_map.shape} for '{title}'"); return
    num_channels_to_plot = int(num_channels) if num_channels is not None else 6
    os.makedirs(plot_dir, exist_ok=True)
    if num_channels_to_plot <=0: return
    relevance_map_cpu = relevance_map.detach().cpu()
    num_ch_available = relevance_map_cpu.shape[0]
    num_ch_to_actually_plot = min(num_channels_to_plot, num_ch_available)
    if num_ch_to_actually_plot == 0: return
    fig, axes = plt.subplots(1, num_ch_to_actually_plot, figsize=(num_ch_to_actually_plot*3, 3.5))
    if num_ch_to_actually_plot == 1: axes = [axes]
    fig.suptitle(title, fontsize=14); vmin,vmax=relevance_map_cpu.min(),relevance_map_cpu.max(); cmap_to_use='hot'
    if vmin >= -1e-5 and vmin <= 1e-5 and vmax >= (1.0-1e-5) and vmax <= (1.0+1e-5) and torch.all((relevance_map_cpu==0)|(relevance_map_cpu==1)): cmap_to_use='gray'
    for i in range(num_ch_to_actually_plot):
        ax=axes[i]; im=ax.imshow(relevance_map_cpu[i],cmap=cmap_to_use,interpolation='nearest',vmin=vmin,vmax=vmax)
        ax.set_title(f'Ch {i}',fontsize=10); ax.axis('off')
    fig.colorbar(im,ax=axes,shrink=0.7,aspect=15,location='right' if num_ch_to_actually_plot>1 else 'bottom')
    plt.tight_layout(rect=[0,0,0.95 if num_ch_to_actually_plot > 1 else 1,0.95])
    safe_title = title.replace(' ', '_').replace('%','pct').replace('~','approx').replace('(','').replace(')','').replace(',','').replace('.','_dot_')
    safe_title = "".join(c for c in safe_title if c.isalnum() or c in ('_','-'))
    filename=os.path.join(plot_dir,f"{safe_title}.png");
    try: plt.savefig(filename); print(f"Saved plot: {filename}")
    except Exception as e: print(f"Error saving plot {filename}: {e}")
    plt.close(fig)

# ============================================
# --- Main Execution Function (VOC/DeepLab CI) ---
# ============================================
def main_voc_ci(config):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")

    print(f"Using device: {device} for VOC DeepLab CI")
    dataloaders, _ = load_voc_data_for_ci(
        config.get("data_dir_voc", DATA_DIR_VOC), # Use config or global default
        config['batch_size_train'],
        INPUT_SIZE_HW_VOC,
        config['num_workers']
    )
    trainloader = dataloaders['train']; testloader = dataloaders['val']
    
    fine_tuned_checkpoint = config.get('fine_tuned_model_path')
    if not os.path.exists(fine_tuned_checkpoint):
        print(f"ERROR: Fine-tuned checkpoint not found at {fine_tuned_checkpoint}"); exit()
        
    model_ci = DeepLabCI_VOC(
        num_classes=NUM_VOC_CLASSES,
        pretrained_checkpoint_path=fine_tuned_checkpoint,
        split_layer_name_str=config['split_layer_name_voc'], # <--- CORRECTED KEYWORD
        model_arch=config.get('model_arch_voc', 'deeplabv3_resnet50'),
        aux_loss_at_load=config.get('aux_loss_at_load_voc', True),
        input_size_hw=(256,256)
    ).to(device)
    print('MODEL:', model_ci)
    print("\n--- Verifying Split Layer Information ---")
    split_module_in_base = model_ci._get_module_by_path(model_ci.base_deeplab_model, model_ci.split_layer_name_str)
    if split_module_in_base is None:
        print(f"DEBUG WARNING: Split layer module '{model_ci.split_layer_name_str}' could not be directly fetched from base_deeplab_model using _get_module_by_path.")
    else:
        print(f"Split layer module type in base_deeplab_model ('{model_ci.split_layer_name_str}'): {type(split_module_in_base)}")
    
    # This is the LRP target name that will be used if LRP masking is chosen.
    # It's set in DeepLabCI_VOC.__init__ to be the same as split_layer_name_str.
    print(f"LRP target layer in base model (from model_ci.lrp_target_layer_in_base): {model_ci.lrp_target_layer_in_base}")
    print(f"This should match the chosen split_layer_name: {config['split_layer_name_voc']}")
    
    print("\n--- Determining Intermediate Activation Shape (VOC) ---")
    INTERMEDIATE_SHAPE_VOC = get_intermediate_activation_shape_voc(model_ci, testloader, device, config) # Pass the config dict
    print(f"Determined intermediate activation shape at split ('{config['split_layer_name_voc']}'): {INTERMEDIATE_SHAPE_VOC}")


    # --- Verifying Split Layer Information (Optional Debug) NOW SAFE TO PRINT INTERMEDIATE_SHAPE_VOC ---
    print("\n--- Verifying Split Layer Information ---")
    split_module_in_base = model_ci._get_module_by_path(model_ci.base_deeplab_model, model_ci.split_layer_name_str)
    if split_module_in_base is None:
        print(f"DEBUG WARNING: Split layer module '{model_ci.split_layer_name_str}' could not be directly fetched from base_deeplab_model using _get_module_by_path.")
    else:
        print(f"Split layer module type in base_deeplab_model ('{model_ci.split_layer_name_str}'): {type(split_module_in_base)}")
    
    print(f"LRP target layer in base model (from model_ci.lrp_target_layer_in_base): {model_ci.lrp_target_layer_in_base}")
    print(f"This should match the chosen split_layer_name: {config['split_layer_name_voc']}")
    
    # Now INTERMEDIATE_SHAPE_VOC is defined and can be printed
    print(f"INTERMEDIATE_SHAPE_VOC (as determined from model_ci.intermediate()): {INTERMEDIATE_SHAPE_VOC}")
    # --- End Verification ---

    LRP_TARGET_FOR_MASK_BASE = model_ci.lrp_target_layer_in_base
    
    # --- Baseline Metric ---
    print("\nCalculating Baseline Pixel Accuracy (Full CI Model on VOC Val)...")
    if config["baseline_eval"]:
        model_ci.eval()
        total_pixels_baseline = 0; correct_pixels_baseline = 0
        with torch.no_grad():
            for inputs, targets in tqdm(testloader, desc="Baseline Eval (VOC)"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs_logits = model_ci(inputs) # Uses DeepLabCI_VOC.forward() -> upsampled
                _, preds = torch.max(outputs_logits, 1)
                valid_mask = targets != IGNORE_INDEX_VOC
                correct_pixels_baseline += torch.sum((preds == targets)[valid_mask]).item()
                total_pixels_baseline += torch.sum(valid_mask).item()
        baseline_pix_acc = 100 * correct_pixels_baseline / total_pixels_baseline if total_pixels_baseline > 0 else 0.0
        print(f"Baseline Full Model Pixel Acc (VOC Val): {baseline_pix_acc:.2f}%")

    

    MASKING_METHOD = config['masking_method']
    final_mask = None; MASK_TYPE_STR = "N/A"; CALIB_MASK_SPARSITY_PARAM = None
    avg_lrp_map_norm = None; avg_abs_act_map = None

    # Calibration loader
    cal_subset_size = min(len(trainloader.dataset), config.get('num_calib_samples', 200))
    if len(trainloader.dataset) == 0: raise ValueError("Training dataset is empty for calibration.")
    cal_indices = np.random.choice(len(trainloader.dataset), cal_subset_size, replace=False).tolist()
    cal_dataset = torch.utils.data.Subset(trainloader.dataset, cal_indices)
    cal_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=config['batch_size_calib'], shuffle=False, num_workers=config['num_workers'])
    print(f"Using {len(cal_dataset)} samples for calibration/avg activation.")

    # MASK CREATION
    if MASKING_METHOD.startswith("LRP"):
        if not CRP_ZENNIT_AVAILABLE: raise RuntimeError(f"CRP/Zennit unavailable for LRP method: {MASKING_METHOD}.")
        
        # LRP_TARGET_FOR_MASK_BASE IS model_ci.lrp_target_layer_in_base (which is model_ci.split_layer_name_str)
        # This is the name of the layer *within the base_deeplab_model*.
        lrp_map_target_layer = model_ci.lrp_target_layer_in_base 
        
        print(f"\n--- LRP Avg Relevance ({MASKING_METHOD}) for layer '{lrp_map_target_layer}' in base model ---")
        if os.path.exists(f"avg_relevance_map_classwise.pt"):
            avg_lrp_map_norm = torch.load(f"avg_relevance_map_classwise.pt")
        else:
            avg_lrp_map_norm = compute_average_relevance_voc(
                model_ci.base_deeplab_model, # LRP on the full, unsplit base model
                cal_loader,
                lrp_map_target_layer, # Layer name relative to base_deeplab_model
                device
            )
        if avg_lrp_map_norm is None: raise RuntimeError(f"Failed LRP map for {MASKING_METHOD}.")
        
        if avg_lrp_map_norm.shape != INTERMEDIATE_SHAPE_VOC:
            # This is the critical check now. No resizing.
            raise RuntimeError(
                f"CRITICAL SHAPE MISMATCH FOR LRP MASKING: "
                f"LRP map for '{lrp_map_target_layer}' has shape {avg_lrp_map_norm.shape}, "
                f"but client intermediate output ('{model_ci.split_layer_name_str}') has shape {INTERMEDIATE_SHAPE_VOC}. "
                "These MUST match. Check splitting logic in DeepLabCI_VOC, the intermediate() method, "
                "and ensure get_intermediate_activation_shape_voc is accurate."
            )
        base_map_for_mask_logic = avg_lrp_map_norm
    
    elif MASKING_METHOD == "MagnitudeBinary":
        avg_abs_act_map = compute_average_abs_activations(model_ci, cal_loader, device) # Shape is INTERMEDIATE_SHAPE_VOC
        base_map_for_mask_logic = avg_abs_act_map
    elif MASKING_METHOD == "RandomBinary":
        base_map_for_mask_logic = None # Not needed
    else:
        raise ValueError(f"Invalid MASKING_METHOD setup: {MASKING_METHOD}")

    # Create final mask based on method
    if MASKING_METHOD == "LRPWeighted":
        final_mask = base_map_for_mask_logic.clone(); MASK_TYPE_STR="LRP Weighted"
    elif MASKING_METHOD == "LRPCalibratedValueKeep" or MASKING_METHOD == "LRPCalibratedBinary":
        is_binary_mask = (MASKING_METHOD == "LRPCalibratedBinary")
        TGT_ACT_SPARSITY=config['target_activation_sparsity_calibrated']
        if os.path.exists(f"final_mask.pt"):
            final_mask = torch.load(f"final_mask.pt")
        else:
            CALIB_MASK_SPARSITY_PARAM = calibrate_value_mask_for_activation_sparsity(
                model_ci, cal_loader, base_map_for_mask_logic, TGT_ACT_SPARSITY, device,
                is_map_binary_creation_logic=is_binary_mask,
                tolerance=config['calib_tolerance'],max_iterations=config['calib_max_iter'], search_step=config.get('calib_step', 5.0)
            )
            final_mask = create_thresholded_mask_from_map(base_map_for_mask_logic, CALIB_MASK_SPARSITY_PARAM, return_binary_indicator=is_binary_mask)
            torch.save(final_mask, f"final_mask.pt")
        MASK_TYPE_STR = f"{MASKING_METHOD.replace('LRP','')} (~{TGT_ACT_SPARSITY:.0f}% Act Sparsity)"
    elif MASKING_METHOD == "MagnitudeBinary":
        TGT_MASK_SPARSITY = config['target_mask_sparsity_direct']
        final_mask = create_thresholded_mask_from_map(base_map_for_mask_logic, TGT_MASK_SPARSITY, return_binary_indicator=True)
        MASK_TYPE_STR=f"Magnitude Binary ({TGT_MASK_SPARSITY:.0f}% Mask Sparsity)"
    elif MASKING_METHOD == "RandomBinary":
        TGT_MASK_SPARSITY = config['target_mask_sparsity_direct']
        # Create on CPU, will be moved to device later if needed
        final_mask = create_random_binary_mask(INTERMEDIATE_SHAPE_VOC, TGT_MASK_SPARSITY, device='cpu')
        MASK_TYPE_STR=f"Random Binary ({TGT_MASK_SPARSITY:.0f}% Mask Sparsity)"

    if final_mask is None: raise RuntimeError("Final mask not created.")
    final_mask = final_mask.cpu() # Ensure it's on CPU
    # Ensure final_mask has the same C, H, W as INTERMEDIATE_SHAPE_VOC
    if final_mask.shape != INTERMEDIATE_SHAPE_VOC:
        raise RuntimeError(f"Final mask shape {final_mask.shape} does not match INTERMEDIATE_SHAPE_VOC {INTERMEDIATE_SHAPE_VOC} after creation!")
    print(f"Final mask generated: shape {final_mask.shape}, Sparsity: {100*count_zeros(final_mask)/final_mask.numel():.2f}% ({MASK_TYPE_STR})")

    # Server Fine-tuning
    if config['do_finetuning_voc']:
        if config['server_ft_model_path'] and os.path.exists(config['server_ft_model_path']):
            model_ci.load_state_dict(torch.load(config['server_ft_model_path'], weights_only=True))
        else:
            model_ci = finetune_server_voc(model_ci, trainloader, final_mask, device, config['finetune_lr_voc'], config['finetune_epochs_voc'])
            torch.save(model_ci.state_dict(), config['server_ft_model_path'])
        MASK_TYPE_STR += " + Finetuned"
    else: MASK_TYPE_STR += " (No Finetuning)"
    
    final_mask_dev = final_mask.to(device)

    # Final Evaluation
    print(f"\n--- Starting Final Evaluation ({MASK_TYPE_STR}) on VOC Val ---")
    model_ci.eval()
    total_pixels_final = 0; correct_pixels_final = 0; batch_act_sparsities_eval = []

    cr_res_file = 'compression_results.csv'
    with open(cr_res_file, mode='w', newline='') as f: # Write header
        writer = csv.writer(f)
        writer.writerow(['Original [bytes]', 'NNCodec [bytes]', 'CR [%]'])

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc=f"Final Eval ({MASK_TYPE_STR})"):
            inputs, targets = inputs.to(device), targets.to(device)
            act_intermediate = model_ci.intermediate(inputs) # [B, C_inter, H_inter, W_inter]
            
            if final_mask_dev.shape != act_intermediate.shape[1:]: # Double check shapes before multiply
                 raise RuntimeError(f"Eval Shape Mismatch: Mask {final_mask_dev.shape} vs Intermediate {act_intermediate.shape[1:]}")

            masked_inter = act_intermediate * final_mask_dev.unsqueeze(0)

            # --- NNCodec ---
            if config['use_nnc']:
                approx_param_base = {"parameters": {}, "put_node_depth": {},
                                     "device_id": 0, "parameter_id": {}} if config["nnc_args"]["tca"] else None

                bitstream = encode(masked_inter, args=config["nnc_args"], quantize_only=config["nnc_args"]["quantize_only"],
                                   approx_param_base=approx_param_base)

                masked_inter = torch.tensor(decode(bitstream, tensor_id=config["nnc_args"]["tensor_id"],
                                                       approx_param_base=approx_param_base), device=device)

                with open(cr_res_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([masked_inter.nbytes, len(bitstream), (len(bitstream)/masked_inter.nbytes) * 100])
            # ---------------

            outputs_logits = model_ci.rest(masked_inter) # Should be upsampled by rest()
            
            _, preds = torch.max(outputs_logits, 1)
            valid_eval_mask = targets != IGNORE_INDEX_VOC
            correct_pixels_final += torch.sum((preds == targets)[valid_eval_mask]).item()
            total_pixels_final += torch.sum(valid_eval_mask).item()
            num_el_act=masked_inter.numel(); batch_act_sparsities_eval.append(100.0*count_zeros(masked_inter)/(num_el_act+1e-9))

    final_pix_acc = 100 * correct_pixels_final / total_pixels_final if total_pixels_final > 0 else 0.0
    avg_act_sparsity_eval = np.mean(batch_act_sparsities_eval) if batch_act_sparsities_eval else 0.0
    print(f"\n--- FINAL Results ({MASK_TYPE_STR} on VOC Val) ---")
    print(f"Final Pixel Acc w/ Mask: {final_pix_acc:.2f}%")
    print(f"Actual Sparsity of Applied Mask: {100*count_zeros(final_mask)/final_mask.numel():.2f}%")
    print(f"Measured Avg Activation Sparsity (Test): {avg_act_sparsity_eval:.2f}%")
    if config["baseline_eval"]:
        print(f"(Baseline Full Model Pixel Acc: {baseline_pix_acc:.2f}%)")

    plot_dir_final = config.get('plot_dir', PLOT_DIR_VOC)
    if avg_lrp_map_norm is not None:
        plot_maps(avg_lrp_map_norm, title=f"Avg_LRP_{LRP_TARGET_FOR_MASK_BASE.replace('.','_')}", plot_dir=plot_dir_final, num_channels=config['plot_num_channels'])
    if avg_abs_act_map is not None:
        plot_maps(avg_abs_act_map, title=f"Avg_AbsAct_{model_ci.split_layer_name_str.replace('.','_')}", plot_dir=plot_dir_final, num_channels=config['plot_num_channels'])
    plot_maps(final_mask, title=f"FinalMask_{MASKING_METHOD.replace(' ','_')}", plot_dir=plot_dir_final, num_channels=config['plot_num_channels'])
    print("\n--- VOC CI Workflow Complete ---")

# ============================================
# --- Main Execution Block (Config for VOC CI) ---
# ============================================
if __name__ == "__main__":
    # ... (voc_ci_config and argparse as in your previous working version) ...
    # Default config from your previous script
    voc_ci_config = {
        "use_cuda": True,
        "num_workers": max(1, os.cpu_count() // 2 -1 if os.cpu_count() and os.cpu_count() > 2 else 0),
        "fine_tuned_model_path": '/Users/adm_becking/Downloads/deeplabv3_voc.pth', #'/home/becking/Downloads/deeplabv3_voc.pth', #'/home/hoefler/remote_inference/segmentation/models/deeplabv3_voc.pth',
        "server_ft_model_path": f'{MODEL_SAVE_BASE_VOC}/server_ft_deeplabv3_voc.pt',
        "model_arch_voc": 'deeplabv3_resnet50',
        "aux_loss_at_load_voc": True,
        # "split_layer_name_voc": "backbone.layer4.2.conv3", # Example: Output of ResNet50 Layer4
        # "split_layer_name_voc": "backbone.layer3.5.conv3", # Example: Output of ResNet50 Layer3
        "split_layer_name_voc": "backbone.layer1", # Example: After ASPP, before final convs
        "masking_method": "LRPCalibratedValueKeep", # "LRPCalibratedValueKeep", "MagnitudeBinary", "RandomBinary"
        "baseline_eval": False,
        "target_activation_sparsity_calibrated": 0.8,
        "target_mask_sparsity_direct": 0.8, # For MagnitudeBinary and RandomBinary
        "calib_tolerance": 1.0, "calib_max_iter": 10, "num_calib_samples": 1000, "calib_subset_divisor": 600, # num_calib_samples overrides divisor logic
        "batch_size_calib": 8,
        "do_finetuning_voc": True, "finetune_lr_voc": 3e-5, "finetune_epochs_voc": 1,
        "batch_size_train": 4, "batch_size_test": 4,
        "plot_num_channels": 50,
        "plot_dir": PLOT_DIR_VOC,
        "data_dir_voc": DATA_DIR_VOC, # Added this
        "use_nnc": True, # enables compression pipeline with NNCodec
    }

    if voc_ci_config["use_nnc"]: # Configuration for NNCodec
        voc_ci_config["nnc_args"] = \
            {"qp": -32,  # Main quantization parameter, small values -> fine quantization, large values -> coarse quantization
             "bitdepth": 4,  # Integer-aligned bitdepth for limited precision [1, 31] bit; note: overwrites QPs
             "use_dq": True,  # Enables dependent scalar / Trellis-coded quantization
             "approx_method": "uniform",  # Approximation / Quantization method: 'uniform' or 'codebook'
             "sparsity": 0.0,  # Target unstructured sparsity [0.0, 1.0], mean- & std-based  (default: 0.0)
             "struct_spars_factor": 0.9,  # Gain factor for structured per-channel sparsity (based on channel means); requires sparsity > 0 (default: 0.9)'
             "tca": False,  # Enables Temporal Context Adaptation (TCA)
             "row_skipping": True,  # Enables skipping tensor rows from arithmetic coding that are entirely zero
             "results": f'{BS_DIR_VOC}',  # Directory to save the bitstream
             "tensor_path": None, # path to tensor if saved to disk, e.g. /tensor.pt or ./tensor.npy
             "job_identifier": "SplitXference",  # Unique identifier for the current job
             "verbose": True,  # Verbose output flag
             "tensor_id": '0', # identifier for coded tensor in the bitstream (default: '0')
             "quantize_only": False, # returns only quantized tensor instead of NNC bitstream, e.g., for testing wiht bz2
            }

    parser = argparse.ArgumentParser(description="Split Inference for VOC Segmentation")
    parser.add_argument('--split_layer_name_voc', type=str, help='Split layer name for DeepLabV3 VOC model')
    parser.add_argument('--masking_method', type=str, choices=["LRPWeighted", "LRPCalibratedValueKeep", "LRPCalibratedBinary", "RandomBinary", "MagnitudeBinary"])
    parser.add_argument('--fine_tuned_model_path', type=str, help='Path to fine-tuned .pth model')
    parser.add_argument('--do_finetuning_voc', type=lambda x: (str(x).lower() == 'true'), help='Enable/disable server fine-tuning')
    parser.add_argument('--target_mask_sparsity_direct', type=float, help="Target sparsity for direct mask methods")


    parsed_args = parser.parse_args()
    for key, value in vars(parsed_args).items():
        if value is not None:
            print(f"  Overriding config '{key}' with CLI value: '{value}'")
            voc_ci_config[key] = value
            
    main_voc_ci(voc_ci_config)