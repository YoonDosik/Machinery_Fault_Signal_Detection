# **Machinery Fault Signal Detection with Deep One-Class Classification**

I have implemented Deep SVDD & other one-class classification algorithms in PyTorch

Additionally Deep SVDD is written based on https://github.com/lukasruff/Deep-SVDD-PyTorch

----------------

# **Abstract**

    Fault detection of machinery systems is a fundamental prerequisite to implementing
    condition-based maintenance, which is the most eminent manufacturing equipment system management strategy. To build the fault detection model, one-class classification algorithms have been
    used, which construct the decision boundary only using normal class. For more accurate one-class
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
