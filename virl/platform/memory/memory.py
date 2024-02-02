import os
import shutil
import pickle

from virl.utils import geocode_utils, common_utils, vis_utils


class Memory(object):
    def __init__(self, output_dir, memory_cfg, resume=False):
        """

        The format of memory is:
        memory = {
            obj_id: [view1_info, view2_info, ...]
        }

        Args:
            output_dir:
            memory_cfg:
        """
        self.memory_cfg = memory_cfg
        self.memory = {}
        self.idx = 0
        self.root_dir = output_dir / memory_cfg.PATH
        os.makedirs(self.root_dir, exist_ok=True)
        
        self.memory_ckpt_path = self.root_dir / 'memory.pkl'
        if os.path.exists(self.memory_ckpt_path):
            self.resume_memory()

    def add(self, view, cur_results):
        self.memory[self.idx] = [view]
        view.set_obj_id(self.idx)
        self.idx += 1

        # to save it on disk
        os.makedirs(os.path.join(self.root_dir, str(self.idx - 1)), exist_ok=True)
        self.save_view_to_file(view, self.idx - 1, cur_results)

    def remove(self):
        raise NotImplementedError

    def retrieve_by_geocode(self, refer_view, radius=5):
        # TODO: optimize this to make it run faster.
        refer_geocode = refer_view.geocode
        refer_category = refer_view.category
        candidate_objects = []
        for obj_id, view_list in self.memory.items():
            for candidate in view_list:
                obj_geocode = candidate.geocode
                distance = geocode_utils.calculate_distance_from_geocode(
                    refer_geocode, obj_geocode
                )
                if distance < radius and candidate.category == refer_category:
                    candidate_objects.append(candidate)
                    break

        return candidate_objects

    def add_new_view_to_exist_memory(self, view, obj_id, cur_results):
        # self.memory[obj_id].append(view)
        self.save_view_to_file(view, obj_id, cur_results)

    def save_view_to_file(self, view, obj_id, cur_results):
        view.set_obj_id(obj_id)
        result_image = vis_utils.draw_with_results(view.image, cur_results)
        result_image.save(f'{self.root_dir}/{obj_id}/{view.geocode[0]}_{view.geocode[1]}_{view.heading}_{view.fov}.png')

    def count_category(self):
        count_result = {}
        for obj_id, view_list in self.memory.items():
            category = view_list[0].category
            if category not in count_result:
                count_result[category] = 1
            else:
                count_result[category] += 1

        return count_result

    def get_all_geocodes(self):
        geocode_list = []
        for obj_id, view_list in self.memory.items():
            geocode_list.append(view_list[0].geocode)
        
        return geocode_list

    def get_all_geocodes_by_category(self):
        geocode_by_cate = {}
        for obj_id, view_list in self.memory.items():
            category = view_list[0].category
            if category not in geocode_by_cate:
                geocode_by_cate[category] = [view_list[0].geocode]
            else:
                geocode_by_cate[category].append(view_list[0].geocode)

        return geocode_by_cate

    def save_memory(self):
        result_dict = {
            'memory': self.memory,
            'idx': self.idx
        }

        with open(self.memory_ckpt_path, 'wb') as f:
            pickle.dump(result_dict, f)

    def resume_memory(self):
        with open(self.memory_ckpt_path, 'rb') as f:
            result_dict = pickle.load(f)

        self.memory = result_dict['memory']
        self.idx = result_dict['idx']
