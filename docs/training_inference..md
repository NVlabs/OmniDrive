# Train & inference
## Train
You can train the model following:

```bash
tools/dist_train.sh projects/configs/OmniDrive/mask_eva_lane_det_vlm.py 8 --work-dir work_dirs/mask_eva_lane_det_vlm/
```


## Evaluation
**1. OpenLoop Planning**
Set load_type = ["planning"] in LoadAnnoatationVQATest.
```bash
dict(type='LoadAnnoatationVQATest', 
        base_vqa_path='./data/nuscenes/vqa/val/', 
        base_conv_path='./data/nuscenes/conv/val/',
        base_counter_path='./data/nuscenes/eval_cf/',
        load_type=["planning"], # please don't test all the questions in single test, it requires quite long time
        tokenizer=llm_path, 
        max_length=2048,),
```

Run the evaluation command
```bash
tools/dist_test.sh projects/configs/OmniDrive/mask_eva_lane_det_vlm.py ./work_dirs/mask_eva_lane_det_vlm/latest.pth 8 --format-only
```

Then you can get results under save_path='./results_planning_only/'.
```bash
model = dict(
    type='Petr3D',
    save_path='./results_planning_only/',  #save path for vlm models.
    ...)
```

The final result is obtained by running
```bash
cd ./evaluation
python eval_planning.py
```