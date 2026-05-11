The goal of this short project is to get an estimate of how many images are necessary to train a bounding-boxes object-detection model.

For this we will use the pothole detection dataset from kaggle.com and state-of-the-art models like YOLOv11.

We will prepare the dataset by creating imbricated increasing sets of respective sizes (10, 20, 50, 100, 500) amouting in total to 80% of the dataset, and use the remaining 20% for the testing set.

We will automate the learning process with 50 epochs.

We will compare the restults based on the following metrics : mAP@.5, mAP@.5:.95 and Inference Speed.