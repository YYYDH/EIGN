# EIGN
### Dependencies

- python >= 3.8
- pytorch >= 2.0.0

You can run the following command to install the required environment:

```shell
pip install -r requirements.txt
```

### Dataset

You can download full dataset from [PDBbind](http://www.pdbbind.org.cn/).

### How to run

To train the model, you can run this command:

```
python train.py --cuda YOUR_DEVICE
```

If you want to verify the performance of the model, you can run this command:

```shell
python predict.py
```

https://doi.org/10.5281/zenodo.14580983
