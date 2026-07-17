import os
import subprocess

from core_pkg.common.utils.eval_utils import run_command_in_dir
from config_large import Config
cfg = Config()

# result_dir = '/home/ckd248/data/SCVideo/camera_generation/results/20260225_104708_mmdit copy'
result_dir = '/data2/ckd248/SCVideo/camera_generation/evaluation/gt'
result_dir = '/home/ckd248/data/SCVideo/camera_generation/evaluation/20260312_091009_scannet_gendop_ours'
out, err, ret = run_command_in_dir(['python', '-m', 'src.extraction', f'checkpoint_path={cfg.clatr_ckpt_path}', f'data_dir={result_dir}'], \
                                    'evaluate/CLaTr')
print(out)
print(err)
out, err, ret = run_command_in_dir(['python', '-m', 'src.eval_only', '--pred_path', f"{os.path.join(result_dir, 'preds.npy')}"], \
                                    'evaluate/eval')
print(out)
print(err)