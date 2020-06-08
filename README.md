# A Machine Learning Analysis of CFPB Complaints 

Group Members: Nora Hajjar, Cecile Murray, Erika Tyagi 

This repository contains our final project for CAPP 30255 - Advanced Machine Learning for Public Policy, taken in the Spring of 2020. We use a variety of machine learning techniques to analyze complaints from the [Consumer Financial Protection Bureau's Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/). The main techniques that we use include unsupervised topic modeling, shallow learning classification, deep learning classification, and natural language processing). 

### Repository Structure 
Our repository is organized as follows: 

- The `explore` directory contains exploratory data analysis 
- The `clustering` directory contains the clustering and topic modeling 
- The `shallow` directory contains the shallow learning classification pipeline 
- The `deep` directory contains the deep learning classification pipeline 

Specific instructions for running the analyses are provided within each directory. 

All code contained in this repository was written by team members specifically for this project. PyTorch code to build the neural networks leveraged code written for assignments in this course. 

### Virtual Environment Setup 

If you're using conda, you'll want to deactivate that first with `conda deactivate`. 

Create the virtual environment. (Do this once) 

```
python3 -mvenv venv
```

Next, activate it. Do this when you're working on the project. You'll want to `deactivate` when you're doing something else.

```
source venv/bin/activate
```

Install required packages (Do this the first time, and if you get a missing package error)

```
pip3 install -r requirements.txt
```

Keep `requirements.txt` up to date by updating the list of packages inside it:

```
pip3 freeze > requirements.txt
```
