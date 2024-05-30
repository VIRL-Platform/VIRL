# Getting Started
---

Before you play on any `V-IRL` agents and benchmarks, please make sure you have follow [INSTALL.md](./INSTALL.md) to prepare environments and models.

## 1. Launch UI backend
Launch the UI backend for potential image displaying
```shell
python -m virl.ui.server
```

## 2. Run V-IRL Agents 
- Follow the provided commands to play with various `V-IRL` agents.
- **Important Note**: Since our `V-IRL` bases on online data sources (*i.e.,* Google Map Platform), the obtained data and images would change as time goes by. Also, as the evolution of GPT, the behavior of Agents might be different to our paper.

#### Peng
- Follow the commands below to run Peng.
```shell
cd tools
python launch.py --cfg_file cfgs/peng/peng.yaml
```

- If you want to modify the waypoints, please modify the `WAY_POINT_LANGUAGE` in the config.

#### Aria
- Follow the commands below to run Aria that recommend places using Google Place reviews.
```shell
cd tools
python launch.py --cfg_file cfgs/aria/aria.yaml
```
- Or you can run Aria that recommend places using web search.
```shell
python launch.py --cfg_file cfgs/aria/aria_websearch.yaml
```

- Notice that Aria might recommend different restaurants as Google Map Platform providing different `nearby search` results as time goes by.

#### Vivek
- Follow the commands below to run Vivek estate recommendation.
```shell
cd tools
python launch.py --cfg_file cfgs/vivek/vivek.yaml
```

#### RX399
- Follow the commands below to run RX-399 trash bin counting on New York City,
```shell
cd tools
python launch.py --cfg_file cfgs/rx399/rx399_ny.yaml
```
- or Hong Kong
```shell
python launch.py --cfg_file cfgs/rx399/rx399_hk.yaml
```
- **Note**: We find that the Google Map Platform returns different street view images now. Please refer to these [images](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jhyang13_connect_hku_hk/EjDHXgZ-_EJHh7vKlREKgA8BS5aakbT1s7_li7PmlWVClQ?e=f3oTw8) to check results in our previous attempts.


#### Imani
- Follow the commands below to run Imani `trash bin` & `hydrant` & `bench` distribution statistic on Central Park, New York City.
```shell
cd tools
python launch.py --cfg_file cfgs/imani/imani.yaml
```

- Note: Imani will takes a huge amount of Google Map Platform credits, around `$300-$400`.

- We provide our collected heatmap data [here](https://drive.google.com/drive/folders/17YTC0fCrxbdA--2DPE417uSD1wTlw2bJ?usp=sharing).

#### Hiro
- Follow the commands below to attempt Hiro intentional exploring in Hong Kong street.
```shell
cd tools
python launch.py --cfg_file cfgs/hiro/hiro.yaml
```

#### Local
- In our paper, Ling and Local are collaborative agents, but they can also run individually.
- Follow the commands below to let Local generate navigation instructions according to a question.
```shell
cd tools
# Example of MoMA design store in NYC
python launch.py --cfg_file cfgs/local/local_nyc_case1.yaml
# Example of Apple store in San Francisco
python launch.py --cfg_file cfgs/local/local_sf_iphone.yaml 
```

#### Ling
- Follow the commands below to run Ling navigation according to instruction in the city street.
```shell
cd tools
# Example of MoMA design store in NYC
python launch.py --cfg_file cfgs/ling/ling_nyc_moma.yaml
# Example of Apple store in San Francisco
python launch.py --cfg_file cfgs/ling/ling_sf_applestore.yaml
```
- *Note*: For better reproduction cases in our paper, we provide previous generated route.


#### Diego
- Follow the commands below to get Diego's Itinerary for you.
```shell
cd tools
python launch.py --cfg_file cfgs/diego/diego.yaml
```
- By default, we disable taking user language & status adjustment. To open, please modify the following keys in the [config](../tools/cfgs/diego/diego.yaml) to `True`:
```yaml
USER_INPUT: False # take user input for revise plan or not
USER_STATUS: False # take user status for revise plan or not
```

## 3. V-IRL Benchmark
### 3.1. Download our pre-collected data
- We pre-collect place-centric image data for `V-IRL Place Recognition & VQA Benchmark` and routes for `V-IRL Vision-Language Navigation Benchmark`.

#### 3.1.1 V-IRL Place Recognition & VQA Benchmark data
- Please download our collected data for `V-IRL Place Recognition & VQA benchmark` [here](https://drive.google.com/file/d/1Kdrgami6_zeQ8rlB_bLrvuSJae7CthXy/view?usp=sharing).

- Move the downloaded `.zip` file to `/YOUR_PATH_TO_VIRL/VIRL/data/benchmark/` and then
```shell
mv virl_place_recognition_vqa_data.zip /YOUR_PATH_TO_VIRL/VIRL/data/benchmark/
cd /YOUR_PATH_TO_VIRL/VIRL/data/benchmark/
unzip virl_place_recognition_vqa_data.zip
```

- After prepared, the folder structure should be
``` shell
.
├── benchmark_localization_polygon_area
├── benchmark_polygon_area
├── place_centric_data  # obtained by unzip virl_place_recognition_vqa_data.zip
├── place_types_20.txt
└── place_types.txt
```

#### 3.1.2 V-IRL Place Recognition & VQA Benchmark data
- Please download our collected data for `V-IRL Vision-Language Navigation Benchmark` [full set](https://drive.google.com/file/d/1EfcfWvi-cUQTXeT8W0jkRIpxA3BFnnPn/view?usp=sharing) and [mini set](https://drive.google.com/file/d/1jNpafSnclsfFpwcJbNHfNEiKGyyV2FS-/view?usp=sharing).

- Move the downloaded `.zip` file to `/YOUR_PATH_TO_VIRL/VIRL/data/benchmark/` and then
```shell
mv virl_benchmark_vln_full.zip /YOUR_PATH_TO_VIRL/VIRL/data/benchmark/
mv virl_benchmark_vln_mini.zip /YOUR_PATH_TO_VIRL/VIRL/data/benchmark/
cd /YOUR_PATH_TO_VIRL/VIRL/data/benchmark/
unzip virl_benchmark_vln_full.zip
unzip virl_benchmark_vln_mini.zip
```

- After prepared, the folder structure should be
```shell
.
├── benchmark_localization_polygon_area
├── benchmark_polygon_area
├── collect_vln_routes  # obtained by unzip virl_benchmark_vln_full.zip
├── collect_vln_routes_subset9  # obtained by unzip virl_benchmark_vln_mini.zip
├── place_centric_data  # obtained by unzip virl_place_recognition_vqa_data.zip
├── place_types_20.txt
└── place_types.txt
```

### 3.2. Collect your own data (optional)
- As mentioned in our paper, we create a automatic data curation and annotation pipeline in `VIRL`. If you just want to **test some models** in `V-IRL benchmark`, you can just pass this part. Nevertheless, if you want to collect your own data as `V-IRL benchmark`, you can refer to this section.

#### 3.2.1 V-IRL Place Recognition benchmark
- Here, we first need to collect place-centric images and related place information as follow:
```shell
cd tools
python launch.py --cfg_file cfgs/collect_data/collect_place_centric_data.yaml
```
- The place-centric image and place information will also be used in `V-IRL Place VQA benchmark`.

#### 3.2.2 V-IRL Place VQA benchmark
- First, please make sure you have already collect place-centric images follow the section `3.2.1` . Then, run the following code to generate VQA pairs based on place-centric images and information
```shell
python launch.py --cfg_file cfgs/collect_data/generate_place_vqa_data.yaml
```

#### 3.2.3 V-IRL Place Localization benchmark
- For this benchmark, we do not pre-collect any data but do online image fetching and evaluation, so the detector can use multiple fov and heading angles on each GPS location.

#### 3.2.4 V-IRL Vision-Language Navigation benchmark
- To collect VLN routes, please run the following scripts:
```shell
python scripts/batch_collect_vln_routes.py
``` 


### 3.3. Install & Run benchmarks
- **Important Note**: For each method/model, you should refer to its corresponding repo to prepare the environment. 

#### 3.3.1 V-IRL Place Recognition benchmark
- The benchmarked methods of this benchmark are mainly implemented in
`virl/perception/recognizer/`.

- Take `CLIP` as an example, run the following script to benchmark it:
```shell
python launch.py --cfg_file cfgs/benchmark/recognition/clip.yaml
```

- Configs for other models lies in `tools/cfgs/benchmark/recognition/*`  


#### 3.3.2 V-IRL Place VQA benchmark
- The benchmarked methods of this benchmark are mainly implemented in
`virl/perception/mm_llm/`.
- Take `BLIP2` as an example, run the following script to benchmark it:
```shell
python launch.py --cfg_file cfgs/benchmark/vqa/place_centric_vqa.yaml
```
- Modify the following keys in `cfgs/benchmark/vqa/place_centric_vqa.yaml` to switch between different models:
```yaml
VISION_MODELS:
  MiniGPT4:
    SERVER: http://127.0.0.1:xxxx # modify to your own address
    BEAM_SEARCH: 1
    TEMPERATURE: 1.0

  MiniGPT4Local:
    PATH: /xxx/MiniGPT-4  # modify to your path
    GPU_ID: 0
    CFG_PATH: /xxx/MiniGPT-4/eval_configs/minigpt4_eval.yaml  # modify to your path

  InstructBLIPLocal:
    MODEL_NAME: blip2_t5_instruct
    MODEL_TYPE: flant5xxl

    MIN_LENGTH: 1
    MAX_LENGTH: 250
    BEAM_SIZE: 5
    LENGTH_PENALTY: 1.0
    REPETITION_PENALTY: 1.0
    TOP_P: 0.9
    SAMPLING: "Beam search"

...
PIPELINE:
  ...
  VQA:
    MM_LLM: BLIP2 # modify the model name
```

#### 3.3.3 V-IRL Place Localization benchmark
- The benchmarked methods of this benchmark are mainly implemented in
`virl/perception/detector/`.
- To run single detector `GLIP` as an example
  
```shell
python launch.py --cfg_file cfgs/benchmark/localization/place_loc.yaml
```
- Modify the following keys in `cfgs/benchmark/localization/place_loc.yaml` to switch between different models:
```yaml
...
VISION_MODELS:
  ...
  GLIP_CLIP:
    GLIP:
      SERVER: http://xxx.xxx.xxx.xxx:xxxx  # modify to your address
      THRESH: 0.4
    CLIP:
      SERVER: http://xxx.xxx.xxx.xxx:xxxx  # modify to your address
      THRESH: 0.8
      TEMPERATURE: 100.
  
  GroundingDINO:
    CFG_FILE: /xxx/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py  # modify to your path
    CKPT_FILE: /xxx/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth  # modify to your path
    BOX_THRESHOLD: 0.35
    TEXT_THRESHOLD: 0.25
...
PIPELINE:
  ...
  CHECK_SURROUNDING:
    ...
    DETECT:
      NAME: GLIP  # modify here to switch benchmarked methods
      PROPOSAL_SCORES: 0.55  # useless for GLIP_CLIP
```

- To run all models on all regions with a single command, please run:
```shell
python scripts/batch_benchmark_place_loc.py
```

#### 3.3.4 V-IRL Vision-Language Navigation benchmark
- The benchmarked methods of this benchmark are mainly implemented in
`virl/perception/recognizer/`.

- Take `oracle` as an example
```shell
python launch.py --cfg_file cfgs/benchmark/vln/benchmark_vln_oracle.yaml
```
- To evaluate on different vision models, please fill in the following keys in `cfgs/benchmark/vln/benchmark_vln_oracle.yaml`
```yaml
VISION_MODELS:
  ...
  PaddleOCR:
    DET_MODEL_DIR: /xxx/PaddleOCR/ckpt/ch_PP-OCRv4_det_server_infer  # modify to your path
    REC_MODEL_DIR: /xxx/PaddleOCR/ckpt/ch_PP-OCRv4_rec_server_infer  # modify to your path
    CLS_MODEL_DIR: /xxx/PaddleOCR/ckpt/ch_ppocr_mobile_v2.0_cls_slim_infer  # modify to your path
    USE_ANGLE_CLS: True
    PROMPT: ocr_result_to_recognition_template
    MODEL: gpt-3.5-turbo-0613
  
  EvaCLIP:
    MODEL_NAME: EVA02-CLIP-bigE-14-plus
    MODEL_PATH: /xxx/EVA/EVA-CLIP/rei  # modify to your path

  LLaVA:
    MODEL_PATH: /xxx/LLaVA/llava-v1.5-13b  # modify to your path
    LOAD_8BIT: False
    LOAD_4BIT: False
```
- Configs for other models lies in `tools/cfgs/benchmark/vln/*`.

- To run single models (`oracle` as an example here) on all regions with a single command, please run:
```shell
# mini set
python scripts/batch_benchmark_vln.py \
--split_file ../data/benchmark/benchmark_polygon_area/split_list_9.txt \
--route_dir_base ../data/benchmark/collect_vln_routes_subset9 \
--cfg_file cfgs/benchmark/vln/benchmark_vln_oracle.yaml
# full set
python scripts/batch_benchmark_vln.py \
--split_file ../data/benchmark/benchmark_polygon_area/split_list_14.txt \
--route_dir_base ../data/benchmark/collect_vln_routes \
--cfg_file cfgs/benchmark/vln/benchmark_vln_oracle.yaml
```


### 3.4 Add custom models
Please stay tuned for the tutorials.
