from genericpath import isdir
import splitfolders
import zipfile
import random
import os
import cv2
import re
import numpy as np

class LoadData:   
    def load_dataset(self, path_to_zip_file, dir_to_extract_to, dir_extracted_images, split_rate):
        cwd = os.getcwd()   # current working directory

        # Unzipping files
        if os.path.isdir(dir_extracted_images):
            pass
        else:            
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(dir_to_extract_to)

        # Split data: train/dev
        splitted_data_dir = cwd + "\\splitted_data"
        
        if os.path.isdir(splitted_data_dir):
            pass
        else:    
            splitfolders.ratio(
                dir_extracted_images ,
                output="splitted_data",
                seed=1337,
                ratio=split_rate
            )


        trainig_data = []
        test_data = []
        
        train_path = cwd + "\\splitted_data\\train\\weather"     # path to training data
        val_path = cwd + "\\splitted_data\\val\\weather"         # path to test data

        # Read images as array and resize all of them to a square shape 
        for img in os.listdir(train_path):
            filename = img.split('.')[0]
            category = re.sub(r'[^a-zA-Z]', '', filename)
            try:
                img_array = cv2.imread(os.path.join(train_path, img))
                img_size = 120
                new_array = cv2.resize(img_array, (img_size, img_size))

                trainig_data.append([new_array, category])
            except Exception as e:
                pass
            
        # Reorder randomly because images were alphabetically ordered 
        random.shuffle(trainig_data)

        X_train = []
        classes = []

        for features, label in trainig_data:
            X_train.append(features)
            classes.append(label)

        # Convert to numpy array and reshape y to the right format with labels encoded
        X_train = np.array(X_train)
        y_train = np.unique(classes, return_inverse=True)[1]
        y_train = y_train.reshape((1, y_train.shape[0]))


        # Same for dev data:
        for img in os.listdir(val_path):
            filename = img.split('.')[0]
            category = re.sub(r'[^a-zA-Z]', '', filename)
            try:
                img_array = cv2.imread(os.path.join(val_path, img))
                img_size = 120
                new_array = cv2.resize(img_array, (img_size, img_size))

                test_data.append([new_array, category])
            except Exception as e:
                pass
            
        random.shuffle(test_data)

        X_test = []
        y_test = []

        for features, label in test_data:
            X_test.append(features)
            y_test.append(label)

        X_test = np.array(X_test)
        y_test = np.unique(y_test, return_inverse=True)[1]
        y_test = y_test.reshape((1, y_test.shape[0]))
        

        return X_train, y_train, X_test, y_test, classes