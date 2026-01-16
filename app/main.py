from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import importlib.util
import sys
from pathlib import Path
from fastapi import Query
from ydata_profiling import ProfileReport 
from fastapi.responses import HTMLResponse
import os
import pandas as pd
import giskard
import importlib
app = FastAPI()

MAX_DATASET_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_FILE_SIZE = 100_000  # 100 KB (reasonable for model.py)

SUBMISSIONS_ROOT = Path("submissions")
SUBMISSIONS_ROOT.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Service is running. Use /upload to send files."}

@app.get("/submissions")
async def list_submissions():
    """
    Returns a list of all existing submission IDs (folder names).
    """
    if not SUBMISSIONS_ROOT.exists():
        return []
    # List only directories within the submissions root
    return [d.name for d in SUBMISSIONS_ROOT.iterdir() if d.is_dir()]

@app.post("/upload/model")
async def upload_model(
    submission_id: str = Query(...),
    model_file: UploadFile = File(...), 
    checkpoint_file: UploadFile = File(...)):
    """
    Args:
        model_file (UploadFile, optional): _description_. Defaults to File(...).
        checkpoint_file (UploadFile, optional): _description_. Defaults to File(...).
    Returns:
       html response with scan reports for each target column. 
    """
    if model_file.filename != "model.py":
        raise HTTPException(status_code=400, detail="File must be named model.py")

    valid_types = {
        "text/plain", 
        "application/octet-stream", 
        "text/x-python", 
        "text/x-script.python",
        "application/x-python-code"
    }
    
    #№if model_file.content_type not in valid_types:
    #    raise HTTPException(status_code=400, detail=f"Invalid file type: {model_file.content_type}")

    contents = await model_file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="model.py too large")
    
    submission_dir = SUBMISSIONS_ROOT / submission_id
    submission_dir.mkdir(exist_ok=True)

    model_path = submission_dir / "model.py"
    model_path.write_bytes(contents)
    
    if checkpoint_file.filename != "checkpoint":
        raise HTTPException(status_code=400, detail="File must be named checkpoint")
    
    checkpoint_contents = await checkpoint_file.read()
    checkpoint_path = submission_dir / "checkpoint"
    checkpoint_path.write_bytes(checkpoint_contents)

    return {
        "status": "uploaded"
    }
    
@app.post("/upload/data")
async def upload_data(
    submission_id: str = Query(...),
    file: UploadFile | None = File(None),
):
    """
    Args:
        file (UploadFile | None, optional): _description_. Defaults to File(None).
    Returns:
        html response with data profiling report.
    """

    if file.filename != "data.csv":
        raise HTTPException(status_code=400, detail="File must be named data.csv")

    contents = await file.read()
    
    if len(contents) > MAX_DATASET_SIZE: 
        raise HTTPException(400, "File too large")
    
    submission_dir = SUBMISSIONS_ROOT / submission_id
    submission_dir.mkdir(exist_ok=True)
    
    dst_dir = submission_dir
    dst_path = dst_dir / file.filename
    dst_path.write_bytes(contents)

    dataframe = pd.read_csv(dst_path)

    for column in dataframe.columns:
        if column[:8] != 'feature_' and column[:7] != 'target_':
            os.remove(dst_path)
            raise HTTPException(status_code=400, detail="All columns must begin with feature_ or target_")
            
    return {
        "status": "uploaded"
    }

@app.get("/check_data")
async def check_data(
    submission_id: str = Query(...)
):
    """
    Calculates data profiling report using ydata-profiling library. Report includes:
    missing values, distributions, correlations, etc.
    """
    submission_dir = SUBMISSIONS_ROOT / submission_id
    if not submission_dir.exists():
        raise HTTPException(status_code=400, detail="Submission ID not found")

    if not os.path.exists(submission_dir / 'data.csv'):
        raise HTTPException(status_code=400, detail="The data.csv must be uploaded before checking the data")

    data = pd.read_csv(submission_dir / 'data.csv')
    profile = ProfileReport(data, title="Data Profiling Report")

    return HTMLResponse(content = profile.to_html())

@app.get("/check_model")
async def check_model(
    submission_id: str = Query(...)
):
    """
    Runs automatic tests listed in https://github.com/Giskard-AI/giskard-oss/tree/main/giskard/scanner,
    which include tag "regression"
    Tests are:
        1. Robustness for perturbations https://github.com/Giskard-AI/giskard-oss/tree/main/giskard/scanner/robustness
            - EU AI Act Article 15
        2. Performance bias https://github.com/Giskard-AI/giskard-oss/tree/main/giskard/scanner/performance. Builds slice using 
            tree algoritm. If the quality for slice (mse) differs from overall performance by a thershold, alert is sent. 
            - EU AI Act Articles 10 & 15
        3. Spurious Correlation https://github.com/Giskard-AI/giskard-oss/blob/main/giskard/scanner/correlation/spurious_correlation_detector.py
            - EU AI Act Article 9
        4. Data leakage https://github.com/Giskard-AI/giskard-oss/blob/main/giskard/scanner/data_leakage/data_leakage_detector.py
            - EU AI Act Article 10
        5. Stochasticity https://github.com/Giskard-AI/giskard-oss/blob/main/giskard/scanner/stochasticity/stochasticity_detector.py
            - EU AI Act Article 15 
        6. Loss based https://github.com/Giskard-AI/giskard-oss/blob/main/giskard/scanner/common/loss_based_detector.py 
            - EU AI Act Articles 9 & 15
    """
    submission_dir = SUBMISSIONS_ROOT / submission_id
    if not submission_dir.exists():
        raise HTTPException(status_code=400, detail="Submission ID not found")
    
    if not os.path.exists(submission_dir / 'data.csv'):
        raise HTTPException(status_code=400, detail="The data.csv must be uploaded before checking the model")

    if not os.path.exists(submission_dir / 'model.py'):
        raise HTTPException(status_code=400, detail="The model.py file must be uploaded before checking the model") 

    if not os.path.exists(submission_dir / 'checkpoint'):
        raise HTTPException(status_code=400, detail="The checkpoint file must be uploaded before checking the model") 
        
    data = pd.read_csv(submission_dir / 'data.csv')
    features_columns = [column for column in data.columns if column[:8] == "feature_"]
    target_columns = [column for column in data.columns if column[:7] == "target_"]
    datasets = [
        giskard.Dataset(data, target = target_columns[i]) for i in range(len(target_columns))
    ]
    os.chdir(submission_dir)
    import model
    importlib.reload(model)
    from model import Model
    model_cur = Model("checkpoint")
    model_cur.load_checkpoint()

    giskard_models = [
        giskard.Model(
        model=lambda x: model_cur.predict(x, i),
        model_type="regression",
        feature_names=features_columns) for i in range(len(target_columns))
    ]

    results = [
        f"<h2>Scan Report for: {target}</h2>" + giskard.scan(giskard_model, dataset).to_html()
        for target, giskard_model, dataset in zip(target_columns, giskard_models, datasets)
    ]
    combined_html = f"""
    <html>
        <head>
            <title>Giskard Batch Scan Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .report-section {{ margin-bottom: 100px; border-top: 2px solid #eee; padding-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Comprehensive Model Scan Report</h1>
            <p>Total targets scanned: {len(target_columns)}</p>
            {"".join([f'<div class="report-section">{report}</div>' for report in results])}
        </body>
    </html>
    """
    
    os.chdir("../..")  # Return to original directory
    return HTMLResponse(content=combined_html)