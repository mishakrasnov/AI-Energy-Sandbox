# AI-Energy-Sandbox

This repository provides an implementation of a robustness and reliability checker for energy forecasting models, aligned with the requirements of the EU AI Act. The model is evaluated using the Giskard module. A detailed description of the implemented tests and their mapping to relevant EU AI Act requirements can be found in `app/main.py`.

---

# Deployment 
To deploy the servie - run the following command; it will be avalible at `http://0.0.0.0:8501`
```bash
docker-compose up --build
```

---

# Using the Dashboard
The Dashboard is orginized by chosing existing submission_id or typying a new one. For now, only xgboost and torch models are avaliable. On the main tab, there is a box to submit checkpoint of a model (xgboost or torch). Examples of model savings for these two types can be found in example_xgboost and example_torch folders. To submit data, choice your dataframe saved on csv file and select target columns. 

---

# How to run checks for model and data
Switch to correspinding tabs and push the button to generate report. 

---

# Demonstration Data

The repository includes example files demonstrating the full workflow. The sample dataset corresponds to gas consumption forecasting for a single household, based on data from:

> [https://www.nature.com/articles/s41597-021-00921-y](https://www.nature.com/articles/s41597-021-00921-y)

This example illustrates how the system can be used to assess energy forecasting models under regulatory-oriented robustness and reliability criteria. The data is presented as `data.csv` and checkpoints for models are stored in example_xgboost and example_torch folders. 

---


# Custom tests (for developers)
For some applications, special tests might be required like evaluating peformance during peak hours or under extreme temperatures. Custom tests are added to `app/main.py` using the following template: 

```python
@giskard.test(
    name="Name",
    tags=["regression", "sanity"]
)
def test_name(model: giskard.Model, dataset: giskard.Dataset):
    # load predictions from the whole dataframe or from slice 
    preds = model.predict(dataset.df)
    
    # test logic 
    
    # metric indicates some meaningfull value associated with test like peformance drop.
    # message desribes the test outcome.

    # test passed
    return giskard.TestResult.passed(
        metric=0,
        message="message",
    )
    
    # error
    return giskard.TestResult.failed(
        metric=0,
        message="message",
    )
```
Custom tests run automaticaly after `model.scan()` call. 
