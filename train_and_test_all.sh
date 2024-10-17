source_model=(vit_base_patch16_224 pit_b_224 cait_s24_224 visformer_small)
target_model=(vit_base_patch16_224 pit_b_224 cait_s24_224 visformer_small deit_base_distilled_patch16_224 tnt_s_patch16_224 levit_256 convit_base)
for source in ${source_model[@]}
do
  python attack.py --source_model $source --gpu 0
  for target in ${target_model[@]}
  do
    python evaluate.py --target_model $target --source_model $source  --gpu 0
  done
  python evaluate_cnn.py --source_model $source  --gpu 0
done