'''

Step 2
Prepare Surrogate Dataset

'''

import os
import random

#Select the best performance victim output
dataset_path = './epoch_460/'
notclean_path = './g_img/epoch_460/'
datatset_length = 600

#Write pathes of 6000 images after cleaning into txt file
def manual_cleared_write():
    dataset_txt = ""
    total = 0
    for f in range(10):
        if os.path.exists(dataset_path + str(f) + '/'):
            f_num = 0
            for file in os.listdir(path = dataset_path + str(f) + '/'):
                if os.path.getsize(dataset_path + str(f) + '/' + file) < 100:
                    print("check this file: "+file)
                if f_num < datatset_length:
                    dataset_txt = dataset_txt + dataset_path + str(f) + '/' + file + " " + str(f) + "\n"
                    f_num += 1
                else:
                    break
            print("Total image num of " + str(f) + " saved : " + str(f_num))
            total = total + f_num

    print("Total number of image: " + str(total))
    with open('./Dataset_Clean.txt','w+') as l_w:
        l_w.write(dataset_txt)
    l_w.close()

#Write pathes of randomly selected 6000 images into txt file
def random_write():
    dataset_txt = ""
    total = 0
    for f in range(10):
        path = notclean_path + str(f) + '/'
        if os.path.exists(path):
            f_num = 0
            files = os.listdir(path)
            random_files = random.sample(files, datatset_length)
            for file in random_files:
                if os.path.getsize(path + file) < 100:
                    print("check this file: "+file)
                source_path = os.path.join(path, file)
                dataset_txt = dataset_txt + source_path + " " + str(f) + "\n"
                f_num += 1
            print("Total image num of " + str(f) + " saved : " + str(f_num))
            total = total + f_num

    print("Total number of image: " + str(total))
    with open('./Dataset.txt','w+') as l_w:
        l_w.write(dataset_txt)
    l_w.close()

#manual_cleared_write()
random_write()