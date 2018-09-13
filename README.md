# Deep-Multi-Instance-Learning
This repo contains implementation code for the deep multi-instance learning (MIL) method proposed within the dissertation entitled: _**Deep Multi-Instance Learning for Automated Pathology Screening in Frontal Chest Radiographs**_. The method is implemented within Python using Tensorflow to express model architectures and faccilitate network training.      

## Usage 
<pre>
NOTE: This implementation code has been tested on a machine runnning python 2.7 and Tensorflow 1.4. 
      However, it should run seamlessly on python 3.x distros and later versions of Tensorflow. 
</pre>

1. Download an image dataset for which _binary classification_ is suitable. For example, the [Kaggle Pneumonia X-ray Collection](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) 
2. Partition the dataset into a _positive_ and a _negative_ class. Place the positive items into `data/datasets/raw/d_1` and the negative items into `data/dataset/raw/d_0`  
3. Prepare the data and launch MIL training by running: `source launch_MIL.sh` from the terminal. 
4. At the conclusion of training, the resulting MIL model, as well as its performance on a test set of the data, is stored `data/checkpoint/stage_2`

## License 
Code within this repository provided under the GPLv3 [License](LICENSE).
