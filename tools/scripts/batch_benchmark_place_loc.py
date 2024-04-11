import os
import tqdm
import argparse


def main(cfg):
    # read file list
    with open('../data/benchmark/benchmark_localization_polygon_area/all_files.txt', 'r') as f:
        file_list = f.readlines()
        file_list = [x.strip() for x in file_list]

    debug_image = False

    # for-loop all files and detectors
    for f in tqdm.tqdm(file_list):
        extra_tag = f.split('/')[-1].split('.')[0][13:]
        file_path = os.path.join('..', 'data', 'benchmark', 'benchmark_localization_polygon_area', f)
        print(extra_tag)
        for d in args.detectors:
            print(f'python launcher.py --cfg_file {cfg.cfg_file} \
                      --extra_tag {extra_tag} --set PIPELINE.NAVIGATION.POLYGON_PATH {file_path} \
                        PIPELINE.CHECK_SURROUNDING.DETECT.NAME {d} PIPELINE.DEBUG_IMAGE {debug_image} {args.set}')
            os.system(f'python launcher.py --cfg_file {cfg.cfg_file} \
                      --extra_tag {extra_tag} --set PIPELINE.NAVIGATION.POLYGON_PATH {file_path} \
                        PIPELINE.CHECK_SURROUNDING.DETECT.NAME {d} PIPELINE.DEBUG_IMAGE {debug_image} {args.set}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/benchmark/localization/place_loc.yaml', help='')
    parser.add_argument('--detectors', type=str, default=['GLIP', 'GLIP_CLIP', 'GroundingDINO', 'OpenSeeD'], help='', nargs='+')
    parser.add_argument('--set', type=str, default='', help='')

    args = parser.parse_args()

    main(args)
