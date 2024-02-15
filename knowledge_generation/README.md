# Emotion Knowledge Generation
This folder contains the source code of our experiments in **ECoK Section 5.1**. Most of the code in this folder comes from the [PeaCoK repository](https://github.com/Silin159/PeaCoK/tree/main/knowledge_generation). The The baseline directory contains code modified from the [alexa-with-dstc9-track1-dataset repository](https://github.com/alexa/alexa-with-dstc9-track1-dataset.git).
## Requirements
You need three enviroments:
1. Python 3.10 for DeBERTa traing, COMET-BART & GPT3.5-turbo & GPT4-turbo, and DeBERTa ranking
2. Python 3.6 for NLG evaluation
3. Python 3.11 for Llama2 inference

We recommend using conda to create the environment
```
conda create -n ecok310 python=3.10
conda activate ecok310
pip install -r requirements310.txt
```
```
conda create -n ecok36 python=3.6
conda activate ecok36
pip install -r requirements36.txt
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
```
```
conda create -n ecok_llama python=3.11
conda activate ecok_llama
pip install -r requirements_llama.txt
```
## Data and Model Download
### Datasets
ECoK data in various formats can be downloaded from [this link](https://drive.google.com/file/d/1MkJh1kqkLAlW3ER0MA8HGoof1CRuiWvW/view?usp=sharing), please unzip and place ECoK.zip to `/data/raw`. The data required for this experiment can be downloaded from [this link](https://drive.google.com/file/d/12gQvQ6xHiDIdUE3uRSl8h_HiOBZcL3XI/view?usp=sharing), please unzip and place the all the data folder to `/data`
### Model Checkpoints
The trained COMET-ECoK is [here](https://drive.google.com/file/d/1OizLrNaBl4_s4UUKAMVNfRKRj3t3zLFI/view?usp=sharing), please move the model checkponts to `/models`. The trained DeBERTa is [here](https://drive.google.com/file/d/1KRn0-JyjjJFBDws-bP_N2i8LOly8-1Ft/view?usp=sharing), move the checkponts to `/deberta_model`
## Model Traing
If you want to train you own COMET-ECoK, please follow the guide [here](https://drive.google.com/file/d/12gQvQ6xHiDIdUE3uRSl8h_HiOBZcL3XI/view?usp=sharing) and replace the dataset to ECoK dataset.

If you want to train you own DeBERTa, run the following script in ecok310 environment:
```
bash deberta_training.sh
```
## Model Inference
If you want to use our ready-generated attributes, you can skip to the Automatic Evaluation section. If you would like to generate the tails yourself, follow the next steps.

To generate tails with COMET-ECoK, run the following script:
```
python tail_generation_comet.py
```
To generate tails with Llama2-13B & Llama2-70B, you need run different versions of Llama2 using the code `llama.py` and run the following script:
```
python tail_generation_llama.py
```
To generate tails with GPT-3.5-Turbo & GPT-4-Trubo, run the following script:
```python
# For GPT-4-trubo
python tail_generation_gpt3.py --openai_api_key=<YOUR_OPENAI_API_KEY> \
    --gpt_model_name="gpt-4-1106-preview"
# For GPT-3.5-trubo
python tail_generation_gpt3.py --openai_api_key=<YOUR_OPENAI_API_KEY> \
    --gpt_model_name="gpt-3.5-turbo-1106"
```
To rank and choose the best generations, edit and run the following script:
```
bash deberta_eval.sh
```
## Automatic Evaluation
For the automatic evaluation, make sure you have the python 3.6 environment activated, edit and run the following script:
```
bash eval.sh
```