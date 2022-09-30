# Social Network Recommendation with GNN

## Dataset-Epinions
As a social platform, it offers a service that allows users to rate items, browse reviews, and add friends to their own "circle". Therefore, the dataset provides a large amount of rating and social information.

## Code Structure
-dataloader.py-load data after preprocess  
-graph.py-draw social network graph  
-main.py-model establishment and evaluation  
-preprocess.py-data preprocessing  
-snowball.py-snowball smapling algorithm to extract sample amount needed  

## Reference
[1] W. Fan et al., "A Graph Neural Network Framework for Social Recommendations," in IEEE Transactions on Knowledge and Data Engineering, vol. 34, no. 5, pp. 2033-2047, 1 May 2022, doi: 10.1109/TKDE.2020.3008732.  
[2] Y. Bai et al., "Efficient Data Loader for Fast Sampling-Based GNN Training on Large Graphs," in IEEE Transactions on Parallel and Distributed Systems, vol. 32, no. 10, pp. 2541-2556, 1 Oct. 2021, doi: 10.1109/TPDS.2021.3065737.  
[3] Chong Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. 2018. Neural Attentional Rating Regression with Review-level Explanations. In Proceedings of the 27th International Conference on World Wide Web. 1583–1592  
[4] Shuiguang Deng, Longtao Huang, Guandong Xu, Xindong Wu, and Zhaohui Wu. 2017. On deep learning for trust-aware recommendations in social networks. IEEE transactions on neural networks and learning systems 28, 5 (2017), 1164–1177  
[5] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems. 1024–1034.
