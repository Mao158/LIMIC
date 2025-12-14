# LIMIC
This repository is the official implementation of the paper 
“Jun-Xiang Mao, Wei Wang, Min-Ling Zhang. Label-Specific Multi-Semantics Metric Learning for Multi-Label Classification. In: Proceedings of the 32th International Joint Conference on Artificial Intelligence (IJCAI'23), Macao, China.”

Github link: <https://github.com/Mao158/LIMIC>
***

## Requirements
- MATLAB 2022b 
- Statistics and Machine Learning Toolbox  12.4
- Bioinformatics Toolbox 4.16.1
- Parallel Computing Toolbox  7.7
***

To start, create a directory of your choice and copy the code there. 

Set the path in your MATLAB to add the directory you just created.

## Demos
This repository provides two demos on **CAL500** and **emotions** multi-label data sets which shows the training and testing phase of the LIMIC. Other data sets is available from [Mulan](http://mulan.sourceforge.net/datasets.html) and [PALM](http://palm.seu.edu.cn/zhangml/Resources.htm#data).

- LIMIC Demo
This demo demonstrates how to implement LIMIC and how to make multi-label classification with LIMIC directly. You can run **"LIMIC_Demo.m"** to do it.

- MLKNN-LIMIC Demo
LIMIC can be utilized to be coupled with similarity/distance-based multi-label classification algorithms. This demo demonstrates the implementation of MLKNN-LIMIC. You can run **"MLKNN_LIMIC_Demo.m"** to do it. Based on this, you can implement other coupling versions of LIMIC according to your requirements.