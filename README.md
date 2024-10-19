# One STEP at a time: Language Agents are Stepwise Planners

## Code for running STEP in ScienceWorld environment.

### Step 1: Install Java
```!apt-get update
!apt-get install openjdk-11-jdk-headless -qq
```
### Step 2: Set up Python environment
```
!pip install -r /content/step/requirements.txt
!pip install openai==0.28

import nltk
import os
nltk.download('punkt')
nltk.download('wordnet')
os.environ['PYTHONPATH'] = "."
os.environ['OPENAI_API_KEY'] = "<-YOUR KEY HERE->"
```
### Step 3: Install Science World environment
```
%cd /content
!git clone https://github.com/allenai/ScienceWorld.git
%cd ScienceWorld
!git checkout exhaustivevalidactions
!git pull
!pip install -e .
```

### Step 4: Get back to STEP repo
```
%cd /content/step
!mkdir -p logs
```
### Step 5: Example command to run STEP in ScienceWorld
```
!python agents/step.py --task-num "29" --var-num 1 --env-step-limit 37 --num-episodes 5 --gpt-model "gpt-4o-mini" --summarize_end_of_episode 1 --device "cpu" --temperature 0.0 --use-last-k-memories 3  --output-path-prefix logs/testrun/ --simplifications-preset easy
```

## Acknowledgement
Parts of the code are modified from [CLIN](https://github.com/allenai/clin.git). We appreciate the authors for making CLIN open-sourced.
