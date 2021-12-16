# Online Voice Transfer

CS 545 FA2021 final project

-------------

# Dependencies
- numpy
- scipy
- matplotlib
- pytorch
- librosa

# Deploy
1. Add directory `saved_models` in root dir.
2. Download VCC 2016 training dataset, decompress into `datasets/vcc2016/`
so that each audio is in `datasets/vcc2016/vcc2016_training/<speaker>/<audio_id>.wav`
3. Run `python scripts/append_train_set.py` to append audios so that
each audio is at least 4 seconds long on average

# Project structure

The main entrance of our code is `source_filter_model_playground.ipynb`
- `models/source_filter_model.py` includes corresponding models and helper 
classes

`model_playground.ipynb` includes some of our past attempts

All the other jupyter notebooks are mostly playgrounds and do not serve for 
demo purposes

# Samples



# TODO

Reorganize the whole project