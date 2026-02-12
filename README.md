# Trajectory Generation with Conservative Value Guidance (TGCVG)

## Acknowledgements
Our code is based on the following code repositories:
- [https://github.com/LAMDA-RL/ADMPO](https://github.com/LAMDA-RL/ADMPO)
- [https://github.com/tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL)
- [https://github.com/Jaewoopudding/GTA](https://github.com/Jaewoopudding/GTA).

Thank you for their amazing work and for sharing their open-source code.

## Requirements
```console
conda env create -n tgcvg python=3.8
conda activate tgcvg
pip install -r requirements.txt
```

## Datasets
Datasets are stored in the data directory. Run the following script to download the datasets and save them in our format:

```console
cd data
python download.sh
```

## Example
The TGCVG training pipeline consists of four main stages:

#### 1.Train Dynamics Model
To train the dynamics model, please run the following command
```
cd dynamic
python main4offline.py --env d4rl --env-name halfcheetah-medium-v2
```
After training, manually move the trained model file to: `src/transformer/dynamic_models/halfcheetah-medium-v2/`.

#### 2.Train Transformer Model
To train the transformer model, please run the following command
```
python src/transformer/train_transformer.py --dataset halfcheetah-medium-v2  --config_name config.yaml
```

#### 3.Augment Trajectory-Level Data
To sample augmented data from trained transformer model, please run the following command
```
python src/transformer/train_transformer.py --dataset halfcheetah-medium-v2  --config_name config.yaml  --load_checkpoint --ckpt_path <ckpt_path> --back_and_forth
```

#### 4.Train Offline RL Algorithm
To train offline RL algorithms with augmented dataset, please run the following command
```
python corl/algorithms/cql.py --env halfcheetah-medium-v2 --GDA TGCVG --seed 0 --max_timesteps 100000 --batch_size 256
```
