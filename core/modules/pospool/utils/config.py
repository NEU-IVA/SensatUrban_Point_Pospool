import yaml
from easydict import EasyDict as edict

config = edict()
# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
config.epochs = 600
config.start_epoch = 1
config.base_learning_rate = 0.01
config.lr_scheduler = 'step'  # step,cosine
config.optimizer = 'sgd'
config.warmup_epoch = 5
config.warmup_multiplier = 100
config.lr_decay_steps = 20
config.lr_decay_rate = 0.7
config.weight_decay = 0
config.momentum = 0.9
config.grid_clip_norm = -1
config.loss_weight = True

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
config.backbone = 'resnet'
config.head = 'resnet_scene_seg'
config.radius = 0.05
config.sampleDl = 0.02
config.density_parameter = 5.0
config.nsamples = [16, 16, 16, 16, 16]
config.npoints = [16, 16, 16, 16, 16]
config.width = 144
config.depth = 2
config.bottleneck_ratio = 2
config.bn_momentum = 0.1

# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #
config.datasets = 'SensatUrban'
config.data_root = ''
config.num_classes = 0
config.num_parts = 0
config.input_features_dim = 3
config.batch_size = 32
config.num_points = 5000
config.num_classes = 40
config.num_workers = 4
# data augmentation
config.x_angle_range = 0.0
config.y_angle_range = 0.0
config.z_angle_range = 0.0
config.scale_low = 2. / 3.
config.scale_high = 3. / 2.
config.noise_std = 0.01
config.noise_clip = 0.05
config.translate_range = 0.2
config.color_drop = 0.2
config.augment_symmetries = [0, 0, 0]
# scene segmentation related
config.in_radius = 2.0
config.num_steps = 500

# ---------------------------------------------------------------------------- #
# io and misc
# ---------------------------------------------------------------------------- #
config.load_path = ''
config.print_freq = 10
config.save_freq = 10
config.val_freq = 10
config.log_dir = 'log'
config.local_rank = 0
config.amp_opt_level = ''
config.rng_seed = 0

# ---------------------------------------------------------------------------- #
# Local Aggregation options
# ---------------------------------------------------------------------------- #
config.local_aggregation_type = 'pospool'  # pospool, continuous_conv
# PosPool
config.pospool = edict()
config.pospool.position_embedding = 'xyz'
config.pospool.reduction = 'sum'
config.pospool.output_conv = False
# adaptive_weight
config.adaptive_weight = edict()
config.adaptive_weight.weight_type = 'dp'  # dp, df, dp_df, fj, dp_fj, fi_df, dp_fi_df, rscnn
config.adaptive_weight.num_mlps = 1
config.adaptive_weight.shared_channels = 1
config.adaptive_weight.weight_softmax = False
config.adaptive_weight.reduction = 'avg'  # sum_conv, max_conv, mean_conv
config.adaptive_weight.output_conv = False
# pointwisemlp
config.pointwisemlp = edict()
config.pointwisemlp.feature_type = 'dp_fj'  # dp_fj, fi_df, dp_fi_df
config.pointwisemlp.num_mlps = 1
config.pointwisemlp.reduction = 'max'
# pseudo_grid
config.pseudo_grid = edict()
config.pseudo_grid.fixed_kernel_points = 'center'
config.pseudo_grid.KP_influence = 'linear'
config.pseudo_grid.KP_extent = 1.0
config.pseudo_grid.num_kernel_points = 15
config.pseudo_grid.convolution_mode = 'sum'
config.pseudo_grid.output_conv = False

config.max_epoch = 600
config.decay_rate = 0.9885531
config.decay_epoch = 1
config.first_subsampling_dl = 0.2
config.activation_fn = 'relu'
config.init = 'xavier'
config.bn_eps = 0.000001
config.weight_decay = 0.001
config.grad_norm = 100
config.in_features_dim = 5
config.bottleneck_ratio = 3
config.first_features_dim = 128
config.local_aggreagtion = 'adaptive_weight'
config.epoch_steps = 500
config.validation_size = 50
config.in_radius = 10.0  # *5
config.augment_scale_anisotropic = True
config.augment_rotation = 'vertical'
config.augment_scale_min = 0.7
config.augment_scale_max = 1.3
config.augment_noise = 0.001
config.augment_color = 0.8

# sensaturban
config.bev_size = 500
config.bev_scale = 0.05
config.bev_range = 12.5
config.num_classes = 13
config.training_size = 33


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                raise ValueError(f"{k} key must exist in config.py")
