# Efficacy of Temporal and Spatial Abstraction for Training Accurate Machine Learning Models: A Case Study in Smart Thermostats

This repository contains the code to run simulations for the "*Efficacy of Temporal and Spatial Abstraction for Training Accurate Machine Learning Models: A Case Study in Smart Thermostats*" paper, submitted in *Energy and Buildings* Journal.
The repository includes the implementation of the temporal and spacial abstraction suggested in the paper as well as the meta-learning based thermal models personalization on a simulated Server environment. The `\mobile` folder contains the implementation of the approach on Android devices.

### Requirements

| Package     | Version |
|-------------|--------|
| python      | 3.10   |
| Tensorflow  | 2.9.1  |
| numpy       | 1.23.3 |

### Data

We use the Ecobee dataset available in [https://bbd.labworks.org/ds/bbd/ecobee](https://bbd.labworks.org/ds/bbd/ecobee) to train personalized thermal models. The dataset should be downloaded to the `/data` folder. Run the `ecobee.py` script for before the first run of the algorithm to generate the preprocessed data.

### Data preprocessing
To clean the data and generate the 6 clusters suggested in the paper for each season, run the following script:
```shell
python ecobee.py
```
### ML Engine

We have implemented the machine learning models using two ML engines. First, using `Tensorflow` for running on the Linux server (used for performance evaluations). Second, using `Numpy` only (`N3`) to support ML training on android devices.
To configure the ML Engine, update the following line in `src/conf.py` 
``
ML_ENGINE = "Tensorflow"  # "N3" or "Tensorflow"
``
**NB:** Android implementation does not support Tensorflow.

## Evaluation

### Configuration

- To select a given cluster for training our of the 6 clusters generated during the preprocessing phase (), set the id of the cluster in one for the main files (`mainP3.py`, `mainFL.py`, `mainCL.py`) as follows:

  ```
  cluster_id = 0
  ```

  To ignore clustering use:

  ```
  cluster_id = None
  ```

The main algorithm parameters are the following:

| Argument     | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| --mp         | Use message passing (MP) via sockets or shared memory (SM)  (default: MP) |
| --rounds     | Number of rounds of collaborative training (default: 500)    |
| --num_users  | Number of peers joining the P2P network (default: 100)       |
| --epochs     | Number of epochs for local training (default: 2)             |
| --batch_size | Batch size (default: 64)                                     |
| --lr         | Learning rate (default: 0.1)                                 |
| --model      | ML model (default: LSTM) LSTM or RNN                         |
| --dataset    | Dataset (default: Ecobee)                                    |

### Execution of the algorithms

To reproduce the experiments of model performance in the paper use the following command:

- To run Centralized Learning (CL)

`python mainCL.py`

- To run Federated Learning (FL)

`python mainFL.py`

- To run Local Learing (LL)

`python mainLL.py`

You can configure every file using the `args` variable.

## Energy Analysis

To perform energy analysis of P3 on the Linux server (Ubuntu 20.04), we developed two methods of energy readings:

- Evaluating the whole program by running the `run.sh` script.
- Evaluating a given method of the algorithm using python decorators.

**NB:** you need to disable virtualization from the bios as we shield the program to one physical core.

### Requirement

We have used the following packages: `powerstat`, `cset-shield`, `cpupower`.

### Energy consumption of the whole program

To measure the energy consumption of the whole program run the following:

`./run.sh -c 0 -p avg -r 1 -d 2 -e "python mainCL.py"`

Run `./run.sh -h` to get a list of the available options and what are used for.

### Energy consumption of a method

To measure the energy consumption of a given method, use the `@measure_energy` decorator.

For example to evaluation the energy consumption of the local learning step, add the following: 

````python
@measure_energy
def local_training(self, device='cpu', inference=True):
	log('event', 'Starting local training ...')
	...
````

End.