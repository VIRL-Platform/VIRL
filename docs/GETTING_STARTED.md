# Getting Started
---

Before you play on any `V-IRL` agents and benchmarks, please make sure you have follow [INSTALL.md](./INSTALL.md) to prepare environments and models.

## Launch UI backend
Launch the UI backend for potential image displaying
```shell
python -m virl.ui.server
```

## Run V-IRL Agents 
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

