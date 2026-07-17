"""Assertions for the promoted static LookCloser Stage-A defaults."""

from nerfstudio.configs.method_configs import method_configs


def test_lookcloser_method_defaults_match_accepted_stage_a() -> None:
    config = method_configs["lookcloser"]
    datamanager = config.pipeline.datamanager
    dataparser = datamanager.dataparser
    sampler = datamanager.pixel_sampler
    model = config.pipeline.model

    assert config.max_num_iterations == 75_941
    assert config.steps_per_eval_batch == 15_188
    assert config.steps_per_eval_image == 15_188
    assert config.steps_per_eval_all_images == 15_188
    assert config.steps_per_save == 15_188
    assert config.save_only_latest_checkpoint is False
    assert dataparser.eval_mode == "filename"
    assert dataparser.scene_scale == 1.5
    assert dataparser.scale_factor == 1.0
    assert dataparser.orientation_method == "up"
    assert dataparser.center_method == "focus"
    assert datamanager.train_num_rays_per_batch == 4096
    assert sampler.enable_fas is True
    assert sampler.fas_strength == 1.0
    assert sampler.sampling_ramp_start == 1.0
    assert sampler.sampling_ramp_end == 3.0
    assert model.eval_num_rays_per_chunk == 2048
    assert model.enable_frequency_grid is True
    assert model.enable_feature_reweighting is True
    assert model.feature_reweighting_strength == 1.0
    assert model.ray_sampling_mode == "adaptive"
    assert model.max_steps_per_ray == 1024
    assert model.adaptive_coarse_step_size == 0.00625
    assert model.adaptive_warmup_steps == 4096
    assert model.occupancy_warmup_steps == 4096
    assert model.occupancy_binary_warmup_steps == 4096
    assert model.stable_occupancy_reduction is True
    assert model.log2_hashmap_size == 23
    assert model.max_res == 8192.0
    assert model.reconstruction_loss_type == "charbonnier"
    assert model.distortion_loss_mult == 0.01
    assert model.background_color == "black"
    assert config.optimizers["fields"]["optimizer"].lr == 0.01
    assert config.optimizers["fields"]["scheduler"].lr_final == 0.0001
    assert config.optimizers["fields"]["scheduler"].max_steps == 200_000


def test_rejected_speed_and_variance_controls_stay_default_off() -> None:
    config = method_configs["lookcloser"]
    pipeline = config.pipeline
    model = pipeline.model

    assert pipeline.target_num_samples_per_batch == 0
    assert pipeline.independent_rng_streams is False
    assert pipeline.tcnn_network_jit_switch_step is None
    assert pipeline.train_rays_switch_step is None
    assert pipeline.feature_reweighting_switch_step is None
    assert pipeline.datamanager.cache_train_rays is False
    assert pipeline.datamanager.cpu_fas_prefetch is False
    assert model.corrected_arm_allocator is False
    assert model.independent_rng_streams is False
    assert model.tcnn_network_jit is False
    assert model.occupancy_diagnostics is True
    assert config.optimizers["fields"]["optimizer"].fused is None
