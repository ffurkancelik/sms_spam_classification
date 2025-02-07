import dataset_helper_functions as dh
from methods import ML_Models
import os

path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(path, 'dataset', 'sms_spam_train.csv')
tf_idf_save_path = os.path.join(path, 'models', 'tf-idf')
trf_save_path = os.path.join(path, 'models', 'transformer')

print("-----------------------------------------------------------")
print("Data Preprocessing Starts for Tf-Idf: \n")
X, y = dh.prepare_data_for_method_ml(file_path, model_path)
print("-----------------------------------------------------------")
print("Classification Models Starts for Tf-Idf: \n")
tf_idf_ml = ML_Models(tf_idf_save_path)
tf_idf_ml.run(X, y)

print("-----------------------------------------------------------")
print("Data Preprocessing Starts for Transformers: \n")
X, y = dh.prepare_data_for_transformer(file_path)
print("-----------------------------------------------------------")
print("Classification Models Starts for Transformers: \n")
trf_ml = ML_Models(trf_save_path)
trf_ml.delete_initial_model('NaiveBayes')
trf_ml.run(X, y)