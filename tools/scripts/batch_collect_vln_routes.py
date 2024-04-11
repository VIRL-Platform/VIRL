import os
import tqdm
import argparse


def main(args):
    polgyon_path_list = []

    # read split file for polygon area
    with open(os.path.join(args.polgyon_path_base, 'split_list_14.txt'), 'r') as f:
        split_file_list = [x.strip() for x in f.readlines()]
        polgyon_path_list = [os.path.join(args.polgyon_path_base, x) for x in split_file_list]

    # for-loop all files and detectors
    for split_path, polygon_area_path in tqdm.tqdm(zip(split_file_list, polgyon_path_list)):
        # get region key
        continent, region = split_path.split('/')
        cur_region_key = f"{region[13:-4]}_{continent}"
    
        cur_output_dir = os.path.join(args.output_dir_base, cur_region_key)
        
        if os.path.exists(cur_output_dir):
            print(f'Output dir {cur_output_dir} exists, skip')
            continue

        print(f'processing {cur_region_key}')

        print(f'python launcher.py --cfg_file cfgs/collect_data/collect_vln_routes.yaml \
                    --extra_tag {cur_region_key} --set PIPELINE.GENERATE_PLACE_QUEUE.POLYGON_PATH {polygon_area_path} \
                    PIPELINE.GENERATE_PLACE_QUEUE.REGION_KEY {cur_region_key} PIPELINE.ROUTE.ROUTE_MODE {args.route_mode}')
        os.system(f'python launcher.py --cfg_file cfgs/collect_data/collect_vln_routes.yaml \
                    --extra_tag {cur_region_key} --set PIPELINE.GENERATE_PLACE_QUEUE.POLYGON_PATH {polygon_area_path} \
                    PIPELINE.GENERATE_PLACE_QUEUE.REGION_KEY {cur_region_key} PIPELINE.ROUTE.ROUTE_MODE {args.route_mode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--polgyon_path_base', type=str, default='../data/benchmark/benchmark_polygon_area', help='')
    parser.add_argument('--output_dir_base', type=str, default='../output/collect_data/collect_vln_routes', help='')
    parser.add_argument('--route_mode', type=str, default='driving', help='')

    args = parser.parse_args()

    main(args)
