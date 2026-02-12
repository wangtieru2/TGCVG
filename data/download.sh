envs='halfcheetah_medium halfcheetah_medium-expert halfcheetah_medium-replay walker2d_medium walker2d_medium-expert walker2d_medium-replay hopper_medium hopper_medium-expert hopper_medium-replay'
for env in $envs
do
    python download_d4rl_datasets.py --env_name=$env&
    wait
done

