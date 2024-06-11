Task 1: run command: python3 assign3_task1.py --image_path ./data_ass3/Task1_3/images --model_name mono+stereo_640x192 --ext png
Task 3: ! source activate monodepth2 && python3 test_simple.py --image_path ./images --model_name /content/monodepth2/content/drive/MyDrive/ftmono+stereo_640x192/models/weights_9 --pred_metric_depth --ext png

! source activate monodepth2 && python3 train.py --model_name ftmono+stereo_640x192 --load_weights_folder ./models/mono+stereo_640x192 --log_dir ./content/drive/MyDrive/ --num_epochs 10 --log_frequency 20 --split small --dataset small --png
