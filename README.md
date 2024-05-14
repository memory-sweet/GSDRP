##data preprocess

python preprocess_mix.py --choice 0

python preprocess_mix_CCLE.py --choice 0


##train

python training_GDSC.py --model 0 --train_batch 256 --val_batch 256 --test_batch 256 --lr 0.0001 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"

python training_CCLE.py --model 0 --train_batch 256 --val_batch 256 --test_batch 256 --lr 0.0001 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"
