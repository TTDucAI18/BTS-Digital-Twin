#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for k, v in vars(self).items():
            key = k[1:] if k.startswith("_") else k
            setattr(group, key, v)
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self.masks = ""
        self._resolution = -1
        self._white_background = False
        # train_test_exp removed: BTS drone data captured in uniform lighting,
        # exposure compensation causes color shift and PSNR degradation on test poses.
        self.data_device = "cuda"
        self.eval = False
        # Hidden competition poses have no images and are needed only by render.py.
        self.skip_test_poses = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 40_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 40_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        # exposure_lr_* removed: no exposure compensation needed for BTS uniform lighting.
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        # 0 keeps the legacy behaviour of resetting throughout densification.
        # Scene profiles can stop resets early so mature distant backgrounds do
        # not keep getting erased while foreground detail is still splitting.
        self.opacity_reset_until_iter = 0
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.00015
        # Gaussians larger than this screen-space radius are pruned after the
        # first opacity reset.  Set to 0 to disable that specific pruning rule.
        self.max_screen_size = 20
        self.max_gaussians = 0
        # Optional staged point budget, e.g. "12000:1500000,24000:4000000".
        # The first cap applies through its iteration; max_gaussians applies
        # afterwards.  This prevents a low gradient threshold from consuming
        # the entire point budget in the first few thousand iterations.
        self.densify_cap_schedule = ""
        # Bound one densification event independently of the global cap. This
        # prevents split/clone temporary tensors from OOMing near the cap.
        # 0 preserves the legacy unlimited event size.
        self.max_new_points_per_densify = 0
        self.foreground_loss_weight = 0.0
        self.foreground_edge_loss_weight = 0.05
        self.image_edge_loss_weight = 0.0
        # Not checkpointed and cannot be inferred for unseen test cameras.
        self.use_exposure_compensation = False
        # Hybrid Depth Scheduler (TASK 2): base weight for DA-v2 depth regularization.
        # Phase 1 (0-5k iters): full strength to anchor Gaussians on BTS tower.
        # Phase 2 (5k-25k iters): linear decay to 0, freeing 3DGS to recover cable geometry.
        # Phase 3 (25k+ iters): depth loss disabled, color-only optimization for fine details.
        # NOTE: Giảm từ 0.1 → 0.02 để không kìm hãm cáp mỏng (COLMAP không đo được depth cáp trên nền trời).
        self.depth_weight_init = 0.02
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
