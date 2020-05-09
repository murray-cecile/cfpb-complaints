# cfpb-complaints
Use NLP machine learning techniques to analyze CFPB consumer complaints


#### Using a virtual environment

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
