# Colorization
Final Project in Advanced Machine Learning 2019

In this repo we looked at the colorization problem from two different angles and compare them in our report. 
 * Using a Generative Adversarial Networks
 * Using the classification approach
 
 
 ## Training
To train a model execute a line like the following but replace `train.py` with the corresponing python file (`train_gan.py` or `train_classification.py`)


 `python train.py -b 32 -p ./stl-10 -m mu -`
 
 
 ## Results
 
 ![classification results](https://github.com/lukas-blecher/Colorization/blob/lab150/figures/classification_good.png?raw=true)
 ![gan_results](https://github.com/lukas-blecher/Colorization/blob/lab150/figures/good-gan-images.png?raw=true)
 
 
 Here are some results where the colorization was successful. 
 Left: Classification, right: GAN
