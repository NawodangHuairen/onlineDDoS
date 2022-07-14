# Filtering DDoS Attacks from Unlabeled Network Traffic Data Using Online Deep Learning

[Online DDoS Filtering](https://dl.acm.org/doi/10.1145/3433210.3453083) investigates how deep learning can be leveraged to filter out application layer attack requests, where ever-changing profiles, the lack of labeled data, and constraints in the online setting pose challenges to online learning mitigation solutions.

We propose two approaches. Our first approach is inspired by statistical methods. Using estimated conditional probability distributions, we adopt transfer learning to apply N on N and M separately and efficiently, combining the results to obtain an online learner. 

![NoverD](/image/PAN2.png)

In our second approach, we formulate a specific loss function more suited for deep learning and use iterative training to solve it in the online setting.

![iterclassifer](/image/enhanced_classifier.png)


## Cite

Please cite our paper if you find this code useful for your own work:

```
@inproceedings{10.1145/3433210.3453083,
	author = {Tann, Wesley Joon-Wie and Tan, Jackie Jin Wei and Purba, Joanna and Chang, Ee-Chien},
	title = {Filtering DDoS Attacks from Unlabeled Network Traffic Data Using Online Deep Learning},
	booktitle = {Proceedings of the 2021 ACM Asia Conference on Computer and Communications Security},
	year = {2021},
}
```


## Prerequisites

Create a Conda environment from the environment.yml file:
```
conda env create -f ddos_env.yml
```


## Data

Our datasets are released on Google Drive.

`https://drive.google.com/drive/folders/1XCvaSv7rcgbkn-WKv0UCmBnemmGd_5Ug?usp=sharing`


## Running the experiments

All code are in the folders:
* /test2_caida07
* /test2_cicFriday
* /test2_cicWed


* To run the N-over-D experiments, execute the scripts in the respective folders.
```
./light_run.sh
```
* Or, to run the enhanced iterative classifier experiments, use the notebooks in each folder.
```
enhanced_classifier.ipynb
```