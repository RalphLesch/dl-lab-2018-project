# Augmentation parameter search and semantic segmentation with fully convolutional networks

* For training first download and extract the training and testing data (described at `data/readme.md`)

* Training must be called with `Train_Net.py`

  `$ python3 Train_Net.py --checkpoint_dir=./checkpoints/ultraslimS_4/ --configuration=4 --aug_type=all`
  Here, `checkpoint_dir` must have the path for the checkpoints folder which stores parameters of the network as training    progresses.
  `configuration` is a flag for the type of decoder to be used.  
  `aug_type` sets the type of the augmentation (None, all, shape, color).  
  Please create a separate checkpoint directory for each individual experiment you run.

* After training the test part will use `Test_Net.py` which will generate a text file with all mean IoUs will be saved at: testIoU.txt

  `$ python3 Test_Net.py --model_path=./checkpoints/ultraslimS=3 --configuration=4`

* You can use the following two scripts as shortcuts to run different configurations with propper folder layout:
  ```sh
  $ ./run_train.sh CHECKPOINT_NAME --aug_type=TYPE
  $ ./run_test.sh CHECKPOINT_NAME
  ```
