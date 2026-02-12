python src/transformer/train_transformer.py --dataset halfcheetah-medium-v2  --config_name config.yaml

# python src/transformer/train_transformer.py --dataset halfcheetah-medium-v2  --config_name config.yaml  --load_checkpoint --ckpt_path $YOUR_CHECKPOINT --back_and_forth

# python corl/algorithms/cql.py --env halfcheetah-medium-v2 --GDA TGCVG --seed 0 --max_timesteps 100000 --batch_size 256