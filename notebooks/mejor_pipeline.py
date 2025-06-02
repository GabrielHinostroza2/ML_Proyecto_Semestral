import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('/Users/luissalamanca/Desktop/Duoc/Machine/ML_Proyecto_Semestral/data/03_features/engineered_data.csv', sep=';', dtype=np.float64)
features = tpot_data.drop('EffectivenessScore', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['EffectivenessScore'], random_state=42)

# Average CV score on the training set was: -7.312352441345042e-05
exported_pipeline = make_pipeline(
    RobustScaler(),
    ElasticNetCV(l1_ratio=0.6000000000000001, tol=0.0001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
