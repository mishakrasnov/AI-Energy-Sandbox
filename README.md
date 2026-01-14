# AI-Energy-Sandbox

This repository provides an implementation of a robustness and reliability checker for energy forecasting models, aligned with the requirements of the EU AI Act. The model is evaluated using the Giskard module. A detailed description of the implemented tests and their mapping to relevant EU AI Act requirements can be found in `app/main.py`.

---

## Deployment

To deploy the service, build and run the Docker container:

```bash
docker build -t fast-report-app .
docker run -p 8000:8000 fast-report-app
```

The service will be available at `http://localhost:8000`.

---

## Data Upload

The input data file *must be named `data.csv` and may contain only two types of columns:

* Columns starting with `feature_` – input features used by the prediction model
* Columns starting with `target_` – retrospective ground-truth target values

To upload the data, run:

```bash
curl -X 'POST' \
  'http://localhost:8000/upload/data' \
  -H 'accept: application/json' \
  -F 'file=@data.csv'
```

---

## Model Upload

The model must be provided as:

* `model.py` – a Python file containing the model class definition
* `checkpoint` – a file containing the pretrained model weights

Upload the model and checkpoint using:

```bash
curl -X 'POST' \
  'http://localhost:8000/upload/model' \
  -H 'accept: application/json' \
  -F 'model_file=@model.py' \
  -F 'checkpoint_file=@checkpoint'
```

---

## Running the Checks

Once both data and model are uploaded, generate the reports using the following endpoints:

* **Model checks**:
  `http://localhost:8000/check_model`

* **Data checks**:
  `http://localhost:8000/check_data`

Each endpoint returns a report evaluating robustness, reliability, and compliance-relevant properties.

---

## Demonstration Data

The repository includes example files demonstrating the full workflow. The sample dataset corresponds to gas consumption forecasting for a single household, based on data from:

> [https://www.nature.com/articles/s41597-021-00921-y](https://www.nature.com/articles/s41597-021-00921-y)

This example illustrates how the system can be used to assess energy forecasting models under regulatory-oriented robustness and reliability criteria.

## Custom tests
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
    
    # metric indicate some meaningfull value associated with test like peformance drop.
    # message desribes the test outcome

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
