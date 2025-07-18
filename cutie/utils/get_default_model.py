"""
A helper function to get a default model for quick testing
"""

import os
from omegaconf import open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra  # 必须导入这个

import torch
from cutie.model.cutie import CUTIE
from cutie.inference.utils.args_utils import get_dataset_cfg
from cutie.utils.download_models import download_models_if_needed


def get_default_model() -> CUTIE:
    gh = GlobalHydra.instance()
    prev_hydra = None
    if gh.is_initialized():
        print("[get_default_model] 检测到已存在的Hydra实例，将进行暂存和恢复。")
        prev_hydra = gh.hydra
        gh.clear()
        print(f"[get_default_model] 已清除现有的Hydra实例: {GlobalHydra.instance()}")
        print(f"[get_default_model] {gh.is_initialized()}")
    try:
        initialize(
            version_base="1.3.2", config_path="../config", job_name="eval_config"
        )
        cfg = compose(config_name="eval_config")

        weight_dir = download_models_if_needed()
        with open_dict(cfg):
            cfg["weights"] = os.path.join(weight_dir, "cutie-base-mega.pth")
        get_dataset_cfg(cfg)

        # Load the network weights
        cutie = CUTIE(cfg).cuda().eval()
        model_weights = torch.load(cfg.weights)
        cutie.load_weights(model_weights)

        return cutie
    finally:
        # 3. 无论成功还是失败，都必须把主程序的Hydra状态恢复原样
        print("[get_default_model] 清理临时上下文并恢复原始Hydra实例...")
        gh.clear()  # 清除我们刚刚创建的临时状态
        if prev_hydra is not None:
            gh.initialize(prev_hydra)  # 恢复主程序的状态
        print("[get_default_model] 上下文恢复完毕。")
