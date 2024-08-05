"""
reference speed.py
"""
import os
import pdb
import onnx
from onnxsim import simplify
import yaml
import torch
from torch import nn
import time
import numpy as np
from src.utils.helper import load_model, concat_feat
from src.config.inference_config import InferenceConfig


def initialize_inputs(batch_size=1, device=torch.device("cpu"), dtype=torch.float16):
    """
    Generate random input tensors and move them to GPU
    """
    feature_3d = torch.randn(batch_size, 32, 16, 64, 64).to(device, dtype=dtype)
    kp_source = torch.randn(batch_size, 21, 3).to(device, dtype=dtype)
    kp_driving = torch.randn(batch_size, 21, 3).to(device, dtype=dtype)
    source_image = torch.randn(batch_size, 3, 256, 256).to(device, dtype=dtype)
    generator_input = torch.randn(batch_size, 256, 64, 64).to(device, dtype=dtype)
    eye_close_ratio = torch.randn(batch_size, 3).to(device, dtype=dtype)
    lip_close_ratio = torch.randn(batch_size, 2).to(device, dtype=dtype)
    feat_stitching = concat_feat(kp_source, kp_driving)
    feat_eye = concat_feat(kp_source, eye_close_ratio)
    feat_lip = concat_feat(kp_source, lip_close_ratio)

    inputs = {
        'feature_3d': feature_3d,
        'kp_source': kp_source,
        'kp_driving': kp_driving,
        'source_image': source_image,
        'generator_input': generator_input,
        'feat_stitching': feat_stitching,
        'feat_eye': feat_eye,
        'feat_lip': feat_lip
    }

    return inputs


def simplify_onnx_model(onnx_model_path):
    # load your predefined ONNX model
    model = onnx.load(onnx_model_path)

    # convert model
    model_simp, check = simplify(model)
    onnx.save(model_simp, onnx_model_path)


class WarpingSpadeModel(nn.Module):
    def __init__(self, warping_module, spade_generator):
        super(WarpingSpadeModel, self).__init__()
        self.warping_module = warping_module
        self.spade_generator = spade_generator

    def forward(self, feature_3d, kp_driving, kp_source):
        # Feature warper, Transforming feature representation according to deformation and occlusion
        dense_motion = self.warping_module.dense_motion_network(
            feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source
        )
        if 'occlusion_map' in dense_motion:
            occlusion_map = dense_motion['occlusion_map']  # Bx1x64x64
        else:
            occlusion_map = None

        deformation = dense_motion['deformation']  # Bx16x64x64x3
        out = self.warping_module.deform_input(feature_3d, deformation)  # Bx32x16x64x64

        bs, c, d, h, w = out.shape  # Bx32x16x64x64
        out = out.view(bs, c * d, h, w)  # -> Bx512x64x64
        out = self.warping_module.third(out)  # -> Bx256x64x64
        out = self.warping_module.fourth(out)  # -> Bx256x64x64

        if self.warping_module.flag_use_occlusion_map and (occlusion_map is not None):
            out = out * occlusion_map

        out = self.spade_generator(out)
        return out


def load_torch_models(cfg, model_config, device=torch.device("cpu"), dtype=torch.float16, is_animal=False):
    """
    Load and compile models for inference
    """
    appearance_feature_extractor = load_model(cfg.checkpoint_F if not is_animal else cfg.checkpoint_F_animal,
                                              model_config,
                                              "cpu", 'appearance_feature_extractor').eval().to(device, dtype=dtype)
    motion_extractor = load_model(cfg.checkpoint_M if not is_animal else cfg.checkpoint_M_animal, model_config, "cpu",
                                  'motion_extractor').eval().to(device, dtype=dtype)

    def custom_forward(self, *args, **kwargs):
        out_dict = self.original_forward(*args, **kwargs)
        return [out_dict[key] for key in ['pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp']]

    motion_extractor.original_forward = motion_extractor.forward
    motion_extractor.forward = custom_forward.__get__(motion_extractor)

    warping_module = load_model(cfg.checkpoint_W if not is_animal else cfg.checkpoint_W_animal, model_config, "cpu",
                                'warping_module').eval().to(device, dtype=dtype)
    spade_generator = load_model(cfg.checkpoint_G if not is_animal else cfg.checkpoint_G_animal, model_config, "cpu",
                                 'spade_generator').eval().to(device, dtype=dtype)
    stitching_retargeting_module = load_model(cfg.checkpoint_S if not is_animal else cfg.checkpoint_S_animal,
                                              model_config,
                                              "cpu",
                                              'stitching_retargeting_module')
    for key in stitching_retargeting_module:
        stitching_retargeting_module[key].to(device, dtype=dtype)
    warping_spade_model = WarpingSpadeModel(warping_module, spade_generator).eval().to(device, dtype=dtype)
    return appearance_feature_extractor, motion_extractor, warping_spade_model, spade_generator, stitching_retargeting_module


if __name__ == '__main__':
    is_animal = True
    device = torch.device("cuda")
    weight_dtype = torch.float16
    onnx_save_dir = "pretrained_weights/liveportrait_animal_onnx" if is_animal else "pretrained_weights/liveportrait_onnx"
    print(onnx_save_dir)
    os.makedirs(onnx_save_dir, exist_ok=True)
    inputs = initialize_inputs(dtype=weight_dtype, device=device)
    """
    input feature_3d shape: torch.Size([1, 32, 16, 64, 64])
    input kp_source shape: torch.Size([1, 21, 3])
    input kp_driving shape: torch.Size([1, 21, 3])
    input source_image shape: torch.Size([1, 3, 256, 256])
    input generator_input shape: torch.Size([1, 256, 64, 64])
    input feat_stitching shape: torch.Size([1, 126])
    input feat_eye shape: torch.Size([1, 66])
    input feat_lip shape: torch.Size([1, 65])
    """
    for key in inputs:
        print(f"input {key} shape:", inputs[key].shape)

    # Load configuration
    cfg = InferenceConfig(device_id=0)
    model_config_path = cfg.models_config
    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # Load and compile models
    appearance_feature_extractor, motion_extractor, warping_module, spade_generator, stitching_retargeting_module = load_torch_models(
        cfg, model_config, is_animal=is_animal, device=device, dtype=weight_dtype)

    # export appearance_feature_extractor
    print("export appearance_feature_extractor >>> ")
    app_outputs = appearance_feature_extractor(inputs['source_image'])
    """
    appearance_feature_extractor output shape: torch.Size([1, 32, 16, 64, 64])
    """
    print("appearance_feature_extractor output shape:", app_outputs.shape)
    torch.onnx.export(
        appearance_feature_extractor,
        (inputs['source_image'],),
        os.path.join(onnx_save_dir, "appearance_feature_extractor.onnx"),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['img'],
        output_names=['output'],
        dynamic_axes=None
    )
    simplify_onnx_model(os.path.join(onnx_save_dir, "appearance_feature_extractor.onnx"))

    # export appearance_feature_extractor
    print("export motion_extractor >>> ")
    motion_outputs = motion_extractor(inputs['source_image'])
    """
    motion_outputs->pitch shape: torch.Size([1, 66])
    motion_outputs->yaw shape: torch.Size([1, 66])
    motion_outputs->roll shape: torch.Size([1, 66])
    motion_outputs->t shape: torch.Size([1, 3])
    motion_outputs->exp shape: torch.Size([1, 63])
    motion_outputs->scale shape: torch.Size([1, 1])
    motion_outputs->kp shape: torch.Size([1, 63])
    """
    for i, key in enumerate(['pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp']):
        print(f"motion_outputs->{key} shape:", motion_outputs[i].shape)
    torch.onnx.export(
        motion_extractor,
        (inputs['source_image'],),
        os.path.join(onnx_save_dir, "motion_extractor.onnx"),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['img'],
        output_names=['pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'],
        dynamic_axes=None
    )
    simplify_onnx_model(os.path.join(onnx_save_dir, "motion_extractor.onnx"))

    # export appearance_feature_extractor
    print("export warping_module >>> ")
    warping_outputs = warping_module(inputs['feature_3d'], inputs['kp_driving'], inputs['kp_source'])
    """
    warping_module->occlusion_map shape: torch.Size([1, 1, 64, 64])
    warping_module->deformation shape: torch.Size([1, 16, 64, 64, 3])
    warping_module->out shape: torch.Size([1, 256, 64, 64])
    """
    # for i, key in enumerate(['occlusion_map', 'deformation', 'out']):
    #     print(f"warping_module->{key} shape:", warping_outputs[i].shape)
    print(f"warping_module output  shape:", warping_outputs.shape)
    # use pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
    torch.onnx.export(
        warping_module,
        (inputs['feature_3d'], inputs['kp_driving'], inputs['kp_source']),
        os.path.join(onnx_save_dir, "warping_spade.onnx"),
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=['feature_3d', 'kp_driving', 'kp_source'],
        output_names=['out'],
        dynamic_axes=None
    )
    simplify_onnx_model(os.path.join(onnx_save_dir, "warping_spade.onnx"))


    def modify_onnx_model(onnx_model_path, onnx_save_path, custom_op_name="GridSample3D"):
        model = onnx.load(onnx_model_path)
        for node in model.graph.node:
            if node.op_type == 'GridSample':
                node.op_type = custom_op_name
        onnx.save(model, onnx_save_path)


    modify_onnx_model(os.path.join(onnx_save_dir, "warping_spade.onnx"),
                      os.path.join(onnx_save_dir, "warping_spade-fix.onnx"))

    # stitching export
    print("export stitching >>> ")
    stitching_model = stitching_retargeting_module['stitching']
    """
    stitching_model output shape: torch.Size([1, 65])
    """
    stitching_outputs = stitching_model(inputs['feat_stitching'])
    print(f"stitching_model output shape:", stitching_outputs.shape)
    torch.onnx.export(
        stitching_model,
        (inputs['feat_stitching'],),
        os.path.join(onnx_save_dir, "stitching.onnx"),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    simplify_onnx_model(os.path.join(onnx_save_dir, "stitching.onnx"))

    # eye_stitching_model export
    print("export eye_stitching_model >>> ")
    eye_stitching_model = stitching_retargeting_module['eye']
    """
    eye_stitching output shape: torch.Size([1, 63])
    """
    eye_stitching_outputs = eye_stitching_model(inputs['feat_eye'])
    print(f"eye_stitching output shape:", eye_stitching_outputs.shape)
    torch.onnx.export(
        eye_stitching_model,
        (inputs['feat_eye'],),
        os.path.join(onnx_save_dir, "stitching_eye.onnx"),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    simplify_onnx_model(os.path.join(onnx_save_dir, "stitching_eye.onnx"))

    # eye_stitching_model export
    print("export lip_stitching_model >>> ")
    lip_stitching_model = stitching_retargeting_module['lip']
    """
    lip_stitching output shape: torch.Size([1, 63])
    """
    lip_stitching_outputs = lip_stitching_model(inputs['feat_lip'])
    print(f"lip_stitching output shape:", lip_stitching_outputs.shape)
    torch.onnx.export(
        lip_stitching_model,
        (inputs['feat_lip'],),
        os.path.join(onnx_save_dir, "stitching_lip.onnx"),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    simplify_onnx_model(os.path.join(onnx_save_dir, "stitching_lip.onnx"))
