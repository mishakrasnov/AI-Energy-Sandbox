# AI-Energy-Sandbox
This repository provides an implementation of a robustness and reliability evaluation for energy forecasting models, aligned with the requirements of the EU AI Act. The model is evaluated using the Giskard module. A detailed description of the implemented tests and their mapping to relevant EU AI Act requirements can be found in `app/main.py`.

---

# Deployment 
To deploy the servie - run the following command; it will be avalible at `http://0.0.0.0:8501`
```bash
docker-compose up --build
```

---

# Using the Dashboard
The Dashboard is orginized by chosing existing submission_id or typying a new one. For now, only xgboost or torch models are avaliable. On the main tab, there is a box to submit checkpoint of a model (xgboost or torch). Examples of models and datasets for these three types can be found in example_xgboost, example_torch folders and in demonstration_preparation.ipynb file . To submit data, choice your dataframe saved as csv file and select target columns. To generate reports of model and data, switch to correspinding tab, configure ieeebus39 if needed and push the button.

---

# Demonstration Data
The repository includes example files demonstrating the full workflow. The sample dataset corresponds to electricity consumption forecasting for a new england zone, based on data from:

> [https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info)

This example illustrates how the system can be used to assess energy forecasting models under regulatory-oriented robustness and reliability criteria. The data and checkpoints for models are stored in example_xgboost and example_torch folders. 