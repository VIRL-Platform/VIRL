import os
import tqdm
import argparse
import pickle

import numpy as np

from pathlib import Path
from prettytable import PrettyTable

from virl.utils import common_utils


class PlaceLocMetric(object):
    def __init__(self, ckpt_dir, detector_name, region_file, place_type_file):
        self.output_dir = Path(ckpt_dir)
        
        self.region_file = region_file
        self.detector_name = detector_name
        self.preds = []
        self.gts = []
        
        self.active_detected_places = {}
        self.active_matched_places = {}
        
        # record results for non-active detection
        # detected places are both located and correctly recognized
        self.detected_places = {}
        # matched places are only located but not necessarily correctly recognized
        self.matched_places = {}

        self.place_types = [line.strip().replace('_', ' ') for line in open(place_type_file, 'r').readlines()]

    def eval_results_all(self):
        with open(self.region_file, 'r') as f:
            regions = f.readlines()

        self.detected_places, self.matched_places, self.place_infos = {}, {}, {}
        self.detected_places_city, self.matched_places_city, self.place_infos_city = {}, {}, {}
        self.detected_places_continent, self.matched_places_continent, self.place_infos_continent = {}, {}, {}
        for region in regions:
            region_name = region.strip().split('/')[-1].split('.')[0][13:]
            ckpt_path = self.output_dir / region_name / \
                self.detector_name / 'checkpoint.pkl'

            result_dict = pickle.load(open(ckpt_path, 'rb'))

            # filter result_dict by n_reviews
            if args.n_reviews > 0:
                result_dict = self.filter_by_n_reivews(result_dict)

            self.detected_places.update(result_dict['detected_places'])
            self.matched_places.update(result_dict['matched_places'])
            self.place_infos.update(result_dict['place_infos'])

            # eval city and continent statistics
            continent, city = common_utils.map_region_to_continent_city(
                region_name[:region_name.find('_mini')] + '_' + region.split('/')[0])
            self.detected_places_city[city] = self.detected_places_city.get(city, 0) + len(result_dict['detected_places'])
            self.matched_places_city[city] = self.matched_places_city.get(city, 0) + len(result_dict['matched_places'])
            self.place_infos_city[city] = self.place_infos_city.get(city, 0) + len(result_dict['place_infos'])
            self.detected_places_continent[continent] = self.detected_places_continent.get(continent, 0) + len(result_dict['detected_places'])
            self.matched_places_continent[continent] = self.matched_places_continent.get(continent, 0) + len(result_dict['matched_places'])
            self.place_infos_continent[continent] = self.place_infos_continent.get(continent, 0) + len(result_dict['place_infos'])

        self.formulate_output()

    def formulate_output(self):
        num_total_place = len(self.place_infos)
        num_detected_place = len(self.detected_places)
        num_matched_place = len(self.matched_places)
        
        table = PrettyTable()
        table.field_names = ['method', '# total_place', '# TP', '# Category-agnostic TP', 'accuracy', 'recall', 'R']
        table.add_row(['non-active', num_total_place, num_detected_place, num_matched_place,
                       f'{num_detected_place / max(num_matched_place, 1e-6):.2f}',
                       f'{num_matched_place / max(num_total_place, 1e-6):.2f}',
                       f'{num_detected_place / max(num_total_place, 1e-6):.2f}'
                       ])
        print(table)

        # export table to csv
        table_file = self.output_dir / 'results_all.csv'
        with open(table_file, 'w', newline='') as f:
            f.write(table.get_csv_string())

        self.formulate_category_output(self.matched_places, self.detected_places)

        self.formulate_specific_output('city', self.matched_places_city, self.detected_places_city, self.place_infos_city)
        self.formulate_specific_output('continent', self.matched_places_continent, self.detected_places_continent, self.place_infos_continent)

    def formulate_category_output(self, matched_places, detected_places):
        table = PrettyTable()
        table.field_names = ['category', '# total', '# TP', '# Category-agnostic TP', 'accuracy', 'recall', 'R']
        
        total = common_utils.count_place_types(self.place_infos, self.place_types)
        matched = {}
        detected = {}
        for place_id in matched_places.keys():
            gt_types = self.place_infos[place_id]['place_types']
            for type in gt_types:
                matched[type] = matched.get(type, 0) + 1

        for result_lists in detected_places.values():
            cur_pred_types = []
            for pred_type, gt_types in result_lists:
                if pred_type not in cur_pred_types:
                    detected[pred_type] = detected.get(pred_type, 0) + 1
                    cur_pred_types.append(pred_type)
        
        accuracy_list = []
        recall_list = []
        r_list = []
        for type in total.keys():
            type_acc = detected.get(type, 0) / matched.get(type, 1e-6)
            type_recall = matched.get(type, 0) / max(total[type], 1e-6)
            type_r = detected.get(type, 0) / max(total[type], 1e-6)
            
            table.add_row([
                type, total[type], detected.get(type, 0),  matched.get(type, 0),
                f'{type_acc:.3f}',
                f'{type_recall:.3f}',
                f'{type_r:.3f}'
            ])
            accuracy_list.append(type_acc)
            recall_list.append(type_recall)
            r_list.append(type_r)
        
        # add average row
        table.add_row([
            'average', np.average(list(total.values())),
            np.average(list(detected.values())), np.average(list(matched.values())),
            f'{np.average(accuracy_list):.3f}',
            f'{np.average(recall_list):.3f}',
            f'{np.average(r_list):.3f}'
        ])
        
        print(table)
        # export table to csv
        table_file = self.output_dir / 'results_category.csv'
        with open(table_file, 'w', newline='') as f:
            f.write(table.get_csv_string())

    def formulate_specific_output(self, category, matched, detected, place_types):
        table = PrettyTable()
        table.field_names = [category, '# total', '# TP', '# Category-agnostic TP', 'accuracy', 'recall']

        for type in place_types.keys():
            table.add_row([
                type, place_types[type], detected.get(type, 0),  matched.get(type, 0),
                f'{detected.get(type, 0) / (matched.get(type, 0) +  + 1e-6):.2f}',
                f'{matched.get(type, 0) / place_types[type]:.2f}'
            ])

        print(table)
        # export table to csv
        table_file = self.output_dir / f'results_{category}.csv'
        with open(table_file, 'w', newline='') as f:
            f.write(table.get_csv_string())

    def filter_by_n_reivews(self, result_dict):
        new_detected_places = {}
        new_matched_places = {}
        new_place_infos = {}

        place_infos = result_dict['place_infos']
        for i, (place_id, place) in enumerate(place_infos.items()):
            if place['n_reviews'] >= args.n_reviews:
                new_place_infos[place_id] = place
                if place_id in result_dict['detected_places']:
                    new_detected_places[place_id] = result_dict['detected_places'][place_id]
                if place_id in result_dict['matched_places']:
                    new_matched_places[place_id] = result_dict['matched_places'][place_id]

        new_result_dict = {
            'step_counter': result_dict['step_counter'],
            'place_infos': new_place_infos,
            'detected_places': new_detected_places,
            'matched_places': new_matched_places
        }
        
        return new_result_dict


def main(args):
    # read file list
    evaluator = PlaceLocMetric(args.ckpt_dir, args.detector, args.region_file, args.place_type_file)
    evaluator.eval_results_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--ckpt_dir', type=str, default="/data/projects/VIRL_internal/output/benchmark/street_detection_no_active", help='')
    parser.add_argument('--region_file', type=str, default='../data/benchmark/benchmark_localization_polygon_area/all_files.txt', help='')
    parser.add_argument('--place_type_file', type=str, default='../data/benchmark/place_types_20.txt', help='')
    parser.add_argument('--detector', type=str, default='GLIP_CLIP', help='')

    parser.add_argument('--n_reviews', type=int, default=-1, help='filter by n_reviews')

    global args
    args = parser.parse_args()

    main(args)
