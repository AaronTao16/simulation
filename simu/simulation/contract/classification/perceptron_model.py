import os

from sklearn.linear_model import SGDClassifier

from simu.simulation.contract.classification.scikit_classifier import SciKitClassifierModule


class PerceptronModule(SciKitClassifierModule):
    def __init__(self, class_weight=None):
        super().__init__(
            _model_initializer=lambda: SGDClassifier(
                loss='perceptron',
                n_jobs=max(1, os.cpu_count() - 2),
                random_state=0xDeCA10B,
                learning_rate='optimal',
                class_weight=class_weight,
                tol=1e-3,
                penalty=None))
