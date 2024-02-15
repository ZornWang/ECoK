# Emotional Reasoning
Code is based on [BHG](https://github.com/SteveKGYang/BHG).

## Data Construction
You can use methods from BHG with our `COMET-ECoK` to  build data, the model checkpoints is [here](https://drive.google.com/file/d/1OizLrNaBl4_s4UUKAMVNfRKRj3t3zLFI/view?usp=sharing). Then put the data in `/comet_enhanced_data`.

## Requirements
Set up the Python 3.7 environment, and build the dependencies. 
```
pip install -r requirements.txt
```

## Training
Five datasets are trained separately. For example:
```
bash IEMOCAP.sh
```

