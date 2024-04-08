import _init_path
import argparse

from pathlib import Path

from virl.config import cfg_from_list, cfg_from_yaml_file, cfg, log_config_to_file
from virl.utils.common_utils import print_stage
from virl.utils import pipeline, common_utils
from tasks import build_task_solver


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger(log_file=None)
    log_config_to_file(cfg, logger=logger)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Initialize
    print_stage('Step 0: Initialize')
    platform, agent, chatbot, messager, custom_agent = pipeline.init_world_and_agent(cfg, output_dir)

    task_solver = build_task_solver(cfg.TASK, output_dir=output_dir, logger=logger)
    task_solver.run(platform, agent, chatbot, messager, args, custom_agent=custom_agent)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--custom_place', type=str, default='None')

    parser.add_argument('--save_image', action='store_true', help='country name')
    parser.add_argument('--resume', action='store_true', help='resume from previous results')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


if __name__ == '__main__':
    main()
