# Online Feature Selection in Intrusion Detection Systems
Most of the feature selection methods for finding the best feature subset are applied in off-line detection mode. In this mini project, I implemeted a feature ranking mechanism for a streaming network traffic data, using weights from linear SVM trained incrementally. The weights are can be sent to the any machine learning box for online-feature selection. 

Related paper can be found at : [Online Feature Ranking for Intrusion Detection Systems!](https://arxiv.org/abs/1803.00530)

### Some properties of the implementation:
* The most important features have the highest (absolute value) weight 
* Features which are critical to detect intrusions (attacks) have negative value
* The best features explaining normal traffic have positive value
* Feature weights change with time and/or attack type
* The computational complexity will decrease in large datasets with the SGD training.

### Results comparing offline and incremental SVM
* Feature weights are used to select best 20 features. Offline SVM is used once in the training, incremental svm is used on-the fly. Neural network with two hidden layers are retrained over chunks of streaming data
* The [ISCX-IDS 2012!](https://www.unb.ca/cic/datasets/index.html) intrusion detection dataset is used for evaluation.
* On-line models quickly adapt to changing behavior in the network data.
* Notable performance improvement over batch learning

![Alt Text](https://github.com/bussfromspace/Online-Feature-Selection-In-Intrusion-Detection-Systems/blob/master/figures/MSEerror.png)
