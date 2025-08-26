# Artificial Intelligence based pathology report generation from gigapixel whole slide images


### WSI Preprocessing
In this work, we use [CLAM](https://github.com/mahmoodlab/CLAM) for preprocessing and feature extraction. 

Request access to [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) to extract features.


## Inference 
## Prerequisite
Create conda environment:
```
git clone https://github.com/PD-pec-101/REG_wsicap
cd REG_wsicap
conda env create -f enviroment.yml
```

### Inference
Download the trained model [wsicap](https://drive.google.com/file/d/1grhI6NU9CyEmqKqKDIM2eRN9W3RVAg46/view) and place this .pt file inside as results/uni2/model_best.pth

To test the model, run:
```
cd REG_wsicap
conda activate wsicap
python main.py --mode 'Test' --image_dir features_test2/uni2/pt_files/ --checkpoint_dir results/uni2/ --save_dir results/uni2 --d_vf 1536 --n_gpu 1
```

To create final json run:
```
python process_json.py --pred_folder results/uni2 
```

## Code 
We have the used the code from [Wsi-Caption](https://github.com/cpystan/Wsi-Caption). We are thankful to the authors for such a useful repository.