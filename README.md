# Act2vec_evaluation
Evaluation of activity embeddings as input for neural networks for Process Mining (inspired by the NLP domain, see for example the famous Word2Vec package: https://code.google.com/archive/p/word2vec/). This code was implemented for my master thesis (I may append the final PDF to the repository as theoretical foundation and discussion of results).

## Setup

- First run the init script (you may need to give permissions) to setup the environment (including Tensorflow GPU support):
```
chmod u+x init.sh
./init.sh
```
- When all necessary libraries are installed and the environment is activated, you can run the main.py file to evaluate word embedding techniques on different datasets (you can adjust the code in main.py to your needs)
```
python3 main.py
```
## Datasets

- BPI Challenge 2012: 10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
- BPI Challenge 2017: 10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
- BPI Challenge 2019: 10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1
- Helpdesk Dataset: 10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb
- Sepsis Datset: 10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460
- Hospital Dataset: 10.4121/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54
