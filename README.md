# Cross-Domain ECG Recognition with Global Refinment of Local Features
***
## Description
This is the code repository for the paper "Cross-Domain ECG Recognition with Global Refinment of Local Features".

## Abstract
Identity details within an ECG is jointly situated within local and global features. The current methods for ECG recognition emphasize only on local or global details. They have also paid limited attention to unseen and cross-domain scenarios. Furthermore, there exists a lack of consensus on evaluation strategies. Thus, this paper introduces LGTraNet, a generalized architecture designed to establish baselines for securing personal identity using ECG biometrics in cross-domain scenarios. Our proposed model firstly extracts identity details at local temporal levels. The extracted features are then calibrated with globally details using a Self-Calibrated Normalizing Residual Network (SCNRNet). Finally, the refined local details are aggregated using a transformer model to formulate robust global identity representations. We evaluate LGTraNet over challenging cross-domain scenarios, such as cross-session and cross-database. To mitigate challenges in domain-shift, we also introduce an incremental learning based training strategy. Experimental study conducted on three benchmark datasets, ECG1D, MIT-BIH, and PTB, shows that the LGTraNet achieves significant performance in cross-domain settings, and outperforms state-of-the-art. Our code is available at \href{https://github.com/AmanVerma2307/LGTraNet.git}{https://github.com/AmanVerma2307/LGTraNet.git

 ![alt text](https://github.com/AmanVerma2307/LGTraNet/blob/master/EBH_LGTraNet.png)
 ![alt text](https://github.com/AmanVerma2307/LGTraNet/blob/master/EBH_SCConv.png)

## Enviroment
```
python: 3.9/3.10
tensorflow: 2.5.0/2.6.0/2.8.0
```
