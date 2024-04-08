# V-IRL Environment Installation 
---

## Environment Preparation 
create a conda environment and install requirements.
```shell
conda create -n virl python=3.8
conda activate virl
pip install -r requirement.txt
```

Build the package
```shell
python setup.py develop
```

## Set API keys
V-IRL relies on Google Map Platform
```shell
export GOOGLE_MAP_API_KEY="YOUR_API_KEY"
export OPENAI_API_KEY="YOUR_API_KEY"
export SERPAPI_API_KEY="YOUR_API_KEY" # optional, used in estate agent vivek only
```

##### For Google Map Platform API key
If you don't have the API key for Google MAP Platform, please visit [Google Map Platform](https://mapsplatform.google.com/) to get one. You would have `$200` credits monthly, which is sufficient for most `V-IRL` exemplar agents.

##### For Estate API key
Please visit [rentcast](https://app.rentcast.io/app) to get the api key. 

## Install Selenium [optional]
In `V-IRL`, we use selenium to control the web browser to interact with the web page using python.
In current version, you should NOT need to explicitly install Selenium. However, if the automatic installation fail, you may need to install it manually as follow:

Take Chrome as an example, you can download suitable driver following [this](https://chromedriver.chromium.org/getting-started). Notice that you have to check your Chrome version and download the corresponding driver.



## Deploy Vision Models [optional]
If you just want to attempt agents that not requiring vision abilities such as: `Peng`, `Aria`, `Vivek` and `Local`, you can just skip this section.

However, if you want to run other agents, you should deploy vision models as follows:

To play `V-IRL` agents in any computer (with or without GPU), we decouple the vision model deployment and V-IRL agents/benchmark running.

**Important Note**: Thanks to our decoupling design, you don't need to install all following vision models in *single* conda environment and machine. 


#### Open-world Detection: GLIP
**Step 1**: Clone and install our custom [GLIP](https://github.com/VIRL-Platform/GLIP). 

**Step 2**: Launch the GLIP server
```shell
python glip_app.py --model_size large
```

**Step 3**: Record the IP and port to the [configure](../tools/cfgs/base_configs/default.yaml) 

```yaml
GLIP:
    ENABLED: True
    ROOT_PATH: ../../GLIP
    MODEL_SIZE: large  # [tiny, large]
    # for GLIP CLIENT ONLY
    SERVER: http://xxx.xxx.xxx.xxx:xxxx # your ip and port here
```


#### Open-world Recognition: CLIP

**Step 1**: Launch CLIP server on any machine
```shell
cd virl/perception/recognizer
python clip_server.py
```

**Step 2**: Record the IP and port to the [configure](../tools/cfgs/base_configs/default.yaml) 
```yaml
CLIP:
    SERVER: http://xxx.xxx.xxx.xxx:xxxx # your ip and port here
```

#### Feature matching: LightGlue
**Step 1**: Clone and install our custom [LightGlue](https://github.com/VIRL-Platform/LightGlue). 

**Step 2**: Launch the LightGlue server
```shell
python app.py
```

**Step 3**: Record the IP and port to the [configure](../tools/cfgs/base_configs/default.yaml) 
```yaml
LIGHTGLUE:
    SERVER: http://xxx.xxx.xxx.xxx:xxxx # your ip and port here
```

