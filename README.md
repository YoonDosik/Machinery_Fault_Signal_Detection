# **Machinery Fault Signal Detection with Deep One-Class Classification**

I have implemented Deep SVDD & other one-class classification algorithms in PyTorch

Additionally Deep SVDD is written based on https://github.com/lukasruff/Deep-SVDD-PyTorch

----------------

# **Abstract**

>> Fault detection of machinery systems is a fundamental prerequisite to implementing condition-based maintenance, which is the most eminent manufacturing equipment system management strategy. 
To build the fault detection model, one-class classification algorithms have been used, which construct the decision boundary only using normal class. For more accurate one-class
classification, signal data have been used recently because the signal data directly reflect the condition
of the machinery system. To analyze the machinery condition effectively with the signal data, features
of signals should be extracted, and then, the one-class classifier is constructed with the features.
However, features separately extracted from one-class classification might not be optimized for the
fault detection tasks, and thus, it leads to unsatisfactory performance. To address this problem, deep
one-class classification methods can be used because the neural network structures can generate
the features specialized to fault detection tasks through the end-to-end learning manner. In this
study, we conducted a comprehensive experimental study with various fault signal datasets. The
experimental results demonstrated that the deep support vector data description model, which is
one of the most prominent deep one-class classification methods, outperforms its competitors and
traditional methods.

----------------

# **Repository structure**

Contains the data. I implemented support for Case Western Reserve University (CWRU) & Paderborn University (Paderborn)

![image](https://github.com/YoonDosik/Machinery_Fault_Signal_Detection/assets/144199897/0fc2b70b-5eb3-4081-93e5-929b4453c5ab)

![image](https://github.com/YoonDosik/Machinery_Fault_Signal_Detection/assets/144199897/19ed14ea-346d-4aa4-a855-7cfc49b843ff)

- CWRU(https://engineering.case.edu/bearingdatacenter)
- Paderborn(https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download)

To utilize for anomaly detection, the datasets must be downloaded from the original sources
