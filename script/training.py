# import the necessary packages
from script.model_localization import pyramid, sliding_window, iou_bbox, non_maximum_supperssion, visualize_bbox
from script.model_classification import SVMObjectClassifier
from sklearn.model_selection import train_test_split
import time
import cv2
import os



def training_model(X, y, feature_extracter, save_path, test_size=0.3, evaluate=False, save_model=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True)
    model = SVMObjectClassifier(C=0.5)
    model.set_feature_extracter(feature_extracter)
    print('Training...')
    model.train(X_train, y_train)
    if save_model:
        print('Saving model...')
        model.save(save_path)
    if evaluate:
        print('Evaluating...')
        model.evaluate(X_test, y_test)
    return model