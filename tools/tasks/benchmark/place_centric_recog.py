import glob
import os
import pickle

import pandas as pd
import PIL.Image as Image
import numpy as np

from tqdm import tqdm

from tools.tasks.task_template import TaskTemplate

from virl.config import cfg
from virl.utils import common_utils
from virl.perception.recognizer.recognizer import Recognizer


class BMPlaceCentricRecognition(TaskTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)
        self.candidates = None  # candidates in string format, separated by ',,'
        self.candidate_list = None  # candidates in list format
        self.cared_labels = None  # identical to candidate_list

        self.image_paths = []
        self.gt_labels = []
        self.place_info = []

        self.top_1_tp = 0
        self.top_3_tp = 0
        self.top_5_tp = 0
        self.total = 0
        
        self.score_list = []  # the score for each label for each image
        self.place_results = {}  # place_id: results

        self.place_info_df = None

    def load_data_and_gt(self, data_cfg):
        image_paths_raw = glob.glob(data_cfg.IMAGE_DIR + '/*.jpg')
        place_info_raw = pickle.load(open(data_cfg.PLACE_INFO, 'rb'))

        assert len(image_paths_raw) == len(place_info_raw)

        for image_path in image_paths_raw:
            place_id = os.path.basename(image_path).split('.')[0]
            place_info = place_info_raw[place_id]

            types = place_info['place_types']
            formulated_types = [type.replace('_', ' ') for type in types]
            categories = common_utils.list_intersection(formulated_types, self.candidate_list)
            # there is no intersection between candidate list and place types
            if len(categories) > 0:
                self.image_paths.append(image_path)
                self.gt_labels.append(categories)
                self.place_info.append(place_info)

    def set_candidates(self, pipeline_cfg):
        candidate_list = []
        with open(pipeline_cfg.RECOGNITION.CANDIDATES_PATH, 'r') as f:
            for line in f:
                candidate_list.append(line.strip().replace('_', ' '))

        self.candidates = ',,'.join(candidate_list)
        self.candidate_list = self.cared_labels = candidate_list

    def record_accuracy(self, ordered_labels, gt_labels):
        top_1 = ordered_labels[0] in gt_labels
        top_3 = len(common_utils.list_intersection(ordered_labels[:3], gt_labels)) > 0
        top_5 = len(common_utils.list_intersection(ordered_labels[:5], gt_labels)) > 0
        
        if top_1:
            self.top_1_tp += 1
        if top_3:
            self.top_3_tp += 1
        if top_5:
            self.top_5_tp += 1
        self.total += 1

        return top_1, top_3, top_5

    @property
    def top_1_acc(self):
        return self.top_1_tp / self.total
    
    @property
    def top_3_acc(self):
        return self.top_3_tp / self.total
    
    @property
    def top_5_acc(self):
        return self.top_5_tp / self.total

    def calc_accs_loop(self):
        # https://github.com/jihanyang/agents/blob/d8cb210ed8ff19dd54c3421b58ad9aeb53817d39/tools/tasks/benchmark/benchmark_place_centric_vqa.py#L164-L182
        mean_acc = {'tp': {}, 'total': {}}
        overall_acc = 0

        for place_id, result in self.place_results.items():
            place_types = result['labels']

            correct = result['top_1']
            overall_acc += correct

            for place_type in place_types:
                mean_acc['tp'][place_type] = mean_acc['tp'].get(place_type, 0) + correct / len(place_types)
                mean_acc['total'][place_type] = mean_acc['total'].get(place_type, 0) + 1 / len(place_types)

        overall_acc /= (len(self.place_results) + 1e-8)
        mean_acc = np.mean([
            place_tp / (mean_acc["total"][place_type] + 1e-8)
            for place_type, place_tp in mean_acc['tp'].items()
        ])

        return overall_acc, mean_acc

    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        pipeline_cfg = cfg.PIPELINE

        # step 1: prepare data
        self.set_candidates(pipeline_cfg)
        self.load_data_and_gt(cfg.PIPELINE.PREPARE_DATA)

        # step 2: init vision model
        print(f'Init vision model: {cfg.PIPELINE.RECOGNITION.NAME}...')
        recognizer = Recognizer(cfg.VISION_MODELS, cfg.PIPELINE.RECOGNITION)

        # step 3: run recognition
        print(f'Running recognition...')
        with tqdm(enumerate(self.image_paths), total=len(self.image_paths)) as pbar:
            for i, image_path in pbar:
                place_id = os.path.basename(image_path).split('.')[0]
                place_labels = self.gt_labels[i]

                if place_id in self.place_results:
                    print(f'Place {place_id} already processed, skipping...')
                    continue

                image = Image.open(image_path)
                result = recognizer.check(image, self.candidates, self.cared_labels)

                ordered_idx = np.argsort(result['scores'])[::-1]
                ordered_labels = result['labels'][ordered_idx]
                top_1, top_3, top_5 = self.record_accuracy(ordered_labels, place_labels)

                # update the description of the progress bar with the current accuracy
                pbar.set_description(
                    f'top-1: {self.top_1_acc:.4f}, ' + \
                    f'top-3: {self.top_3_acc:.4f}, ' + \
                    f'top-5: {self.top_5_acc:.4f}'
                )

                # store the scores for later analysis
                self.score_list.append(result['scores'])

                self.place_results[place_id] = dict(
                    top_1=top_1,
                    top_3=top_3,
                    top_5=top_5,
                    labels=place_labels
                )

        # step 4: calculate accuracy
        print(f'Top-1 Accuracy: {self.top_1_tp}/{self.total} ({self.top_1_acc:.4f})')
        print(f'Top-3 Accuracy: {self.top_3_tp}/{self.total} ({self.top_3_acc:.4f})')
        print(f'Top-5 Accuracy: {self.top_5_tp}/{self.total} ({self.top_5_acc:.4f})')
        
        # step 5: record all predictions
        self.record_results()

        print('Done!')

    def record_results(self):
        # create a df with the scores and labels
        df = pd.DataFrame(self.score_list, columns=self.candidate_list)
        # df['image_path'] = self.image_paths
        df['gt_labels'] = self.gt_labels
        scores_path = self.output_dir / 'scores.csv'
        df.to_csv(scores_path)
        print(f'Scores saved to {scores_path}')

        # use the place info
        info_df = pd.DataFrame(self.place_info).set_index('place_id')
        results_df = pd.DataFrame(self.place_results).T
        joined_df = info_df.join(results_df, how='inner')

        # add continent and city using the region
        joined_df["continent"], joined_df["city"] = zip(*joined_df["region"].apply(common_utils.map_region_to_continent_city))

        # save the results
        aggs = {"top_1": "mean", "top_3": "mean", "top_5": "mean", "name": "count"}

        city_path = self.output_dir / 'city.csv'
        city_df = joined_df.groupby("city").agg(aggs).rename(columns={"name": "count"})
        city_df.reset_index().to_csv(city_path)
        print(f'City results saved to {city_path}')

        continent_path = self.output_dir / 'continent.csv'
        continent_df = joined_df.groupby("continent").agg(aggs).rename(columns={"name": "count"})
        continent_df.reset_index().to_csv(continent_path)
        print(f'Continent results saved to {continent_path}')

        # overall accuracy
        overall_path = self.output_dir / 'overall.csv'
        overall_df = joined_df.agg(aggs).to_frame().T.rename(columns={"name": "count"})
        overall_df.reset_index().to_csv(overall_path)
        print(f'Overall results saved to {overall_path}')

        """Place-level results"""

        # split the scores across each label in the ground truth
        cols = ["top_1", "top_3", "top_5"]
        for col in cols:
            joined_df[f"{col}_split"] = joined_df.apply(lambda row: row[col] / len(row["labels"]), axis=1)

        joined_df["count_split"] = 1 / joined_df["labels"].apply(len)


        # explode the labels into separate rows
        exploded_df = joined_df.explode('labels')
        split_cols = [f"{col}_split" for col in cols]
   
        # pl_aggs = dict(top_1="mean", top_3="mean", top_5="mean", name="sum")
        # pl_aggs = {"top_1_split": "sum", "top_3_split": "sum", "top_5_split": "sum", "count_split": "sum"}
        pl_aggs = {col: "sum" for col in split_cols}
        pl_aggs.update({"count_split": "sum"})

        # overall mean-accuracy (across places)
        # group by place and then average over the places
        overall_ma_path = self.output_dir / 'mean_acc_overall.csv'
        op_agg = exploded_df.groupby("labels").agg(pl_aggs)
        
        # divide each column by the number of times the label appears
        op_agg[split_cols] = op_agg[split_cols].div(op_agg["count_split"], axis=0)
        op = op_agg.drop(columns=["count_split"]).mean()
        op.to_frame().T.to_csv(overall_ma_path)
        print(f'Overall mean-accuracy results saved to {overall_ma_path}')

        op_top_1 = op["top_1_split"]
        _, manual_mean_acc = self.calc_accs_loop()
        assert np.isclose(op_top_1, manual_mean_acc), f"op_top_1: {op_top_1}, manual_mean_acc: {manual_mean_acc}"

        # per city mean-accuracy
        pc_aggs = {col: "mean" for col in split_cols}
        
        # group by city and place and then average over the places
        city_ma_path = self.output_dir / 'mean_acc_city.csv'
        city_agg = exploded_df.groupby(["city", "labels"]).agg(pl_aggs)
        city_agg[split_cols] = city_agg[split_cols].div(city_agg["count_split"], axis=0)
        city_ma_df = city_agg.groupby("city").agg(pc_aggs)
        city_ma_df.reset_index().to_csv(city_ma_path)
        print(f'City mean-accuracy results saved to {city_ma_path}')

        city_place_path = self.output_dir / 'place_per_city.csv'
        city_place_df = city_agg.reset_index()
        city_place_df.to_csv(city_place_path)
        print(f'City place results saved to {city_place_path}')

        # accuracy per place
        place_accuracy = self.output_dir / 'acc_per_place.csv'
        place_accuracy_df = exploded_df.groupby("labels").agg(aggs).rename(columns={"name": "count"})
        place_accuracy_df.reset_index().to_csv(place_accuracy)
        print(f'Place accuracy results saved to {place_accuracy}')
