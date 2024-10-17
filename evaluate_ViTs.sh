echo "The adv_path: $1"
target_model=(vit_base_patch16_224 pit_b_224 cait_s24_224 visformer_small deit_base_distilled_patch16_224 tnt_s_patch16_224 levit_256 convit_base)
for target in ${target_model[@]}
do
  python evaluate.py --target_model $target --source_model $1  --gpu 0
done