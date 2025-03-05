# Object Detection Training and Inferencing Pipeline

## Running the Training Pipeline

### Step 1: Split Data (If Not Already Done)
If you have not split the data into train/test folders, run the following command while in the object_detection subfolder:
```bash
python3 preprocessing/data_split.py
```

### Step 2: Train the model

Navigate to the object_detection subfolder (If Not Already Done), and run execute the following:

```bash
python3 training/transfer_learning.py
```

### Step 3: Move Training Weights (TODO(LM): Automate the process)
Once training is complete, the model weights will be stored in the object_detection folder. Move them to the CNNModels subfolder. (May require to replace the old <b>best.pt</b> file)
```bash
mv object_detection/best.pt CNNModels/
```


### Running Inference
Run the following to run inference
```bash
python3 run/ml_embeddedprod.py
```

### TODO List (LM):

* Store trained CNN model weights in CNNModels subfolder when training is complete
    - Requires replacing the old best.pt file with the new best.pt file
* Store thesholds.yaml file from training in the config folder
* Have the web team send the thresholds.yaml file to the Jetson
    - Consider overwriting config.yaml for that race with the new thresholds before storing and sending to Jetson