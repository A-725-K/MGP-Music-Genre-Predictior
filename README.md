# MGP-Music-Genre-Predictior
Machine Learning project: a *music genre predictor*, based on the SVM algorithm.

## Getting started
The entire project was developed under the GNU Octave platform, relying on the *optim* package, so if you want to clone/download this repository you have to check if both are installaed on your system, too.
To download this repository:
```
git clone https://github.com/A-725-K/MGP-Music-Genre-Predictior.git
```

## How to use
With respect to the test you intend to execute, uncomment only one among these two couple of instruction in the file *main.m* and then you have to simply launch the program.

### Datasets
Both dataset were downloaded from Kaggle, and you can find them [here](https://www.kaggle.com/insiyeah/musicfeatures). The former is a subset of the latter. They have 100 samples each class and 30 features.<br><br> 

**data_2genre.csv** -> smaller dataset with only 2 genres, that is *POP* and *CLASSICAL*.<br>
**data.csv** -> bigger dataset with 10 genres.

```matlab
% --- DATASET 10 GENRES --- %
dataset_name = 'inputs/data.csv';
N = 10; % number of classes
% ------------------------- %

% --- DATASET 2 GENRES --- %
%dataset_name = 'inputs/data_2genre.csv';
%N = 2; % number of classes
% ------------------------ %
```

## Author

* **<i>Andrea Canepa</i>** - Computer Science, UNIGE - *Machine Learning course 2018-2019*
