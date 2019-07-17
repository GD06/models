
python3 trainer.py --logtostderr --batch_size=400 --env=DuplicatedInput-v0 \
    --validation_frequency=25 --tau=0.1 --clip_norm=50 \
    --num_samples=10 --objective=urex --model_name=pcl_rl_urex

#python3 trainer.py --logtostderr --batch_size=400 --env=DuplicatedInput-v0 \
#    --validation_frequency=25 --tau=0.1 --clip_norm=50 \
#    --num_samples=10 --objective=reinforce --model_name=pcl_rl_reinforce
