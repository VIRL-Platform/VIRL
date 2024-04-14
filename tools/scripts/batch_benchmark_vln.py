import os
import tqdm
import argparse


def main(args):
    # read split file for polygon area
    with open(os.path.join(args.split_file), 'r') as f:
        split_file_list = [x.strip() for x in f.readlines()]

    # for-loop all files and detectors
    for split_path in tqdm.tqdm(split_file_list):
        # get region key
        continent, region = split_path.split('/')
        cur_region_key = f"{region[13:-4]}_{continent}"
        cur_route_info_path = os.path.join(args.route_dir_base, cur_region_key, 'route_infos.json')

        print(f'processing {cur_region_key}')

        print(f'python launcher.py --cfg_file {args.cfg_file} \
                    --extra_tag {cur_region_key} --set PIPELINE.DATA.ROUTE_PATH {cur_route_info_path} {args.set}')
        os.system(f'python launcher.py --cfg_file {args.cfg_file} \
                    --extra_tag {cur_region_key} --set PIPELINE.DATA.ROUTE_PATH {cur_route_info_path} {args.set}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--split_file', type=str, default='../data/benchmark/benchmark_polygon_area/split_list_9.txt', help='')
    parser.add_argument('--route_dir_base', type=str, default='../data/benchmark/collect_vln_routes', help='')
    parser.add_argument('--cfg_file', type=str, default='cfgs/benchmark/benchmark_vln/benchmark_vln_l1.yaml', help='')
    parser.add_argument('--set', type=str, default='', help='')
    
    args = parser.parse_args()

    main(args)
