


data_path="data_path"

save_dir="./save_dir"

tmp_save_dir="./tmp_save_dir/"
tsb_dir="./tsbs/affinity_$(date +"%Y-%m-%d_%H-%M-%S")_tsb"

n_gpu=4
MASTER_PORT=10086
finetune_mol_model="./pretrain_weights/mol_pre_no_h_220816.pt"
finetune_pocket_model="./pretrain_weights/pocket_pre_220816.pt"



weight_path="./pretrain_weights/profsa.pt"


batch_size=96
#batch_size=64
batch_size_valid=64


epoch=50
dropout=0.0
warmup=0.06
update_freq=1
dist_threshold=8.0
lr=1e-5

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid,test \
       --num-workers 8 --ddp-backend=c10d \
       --task binding_affinity --loss finetune_mse_single --arch binding_affinity  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_loss --patience 20 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-last-epochs 5 \
       --find-unused-parameters \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --reg \
       --finetune-from-model $weight_path \