import giskard
from typing import Any, List, Optional
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from giskard.scanner.issues import Issue 
import numpy as np

class IssueLevel(str, Enum):
    MAJOR = "major"
    MEDIUM = "medium"
    MINOR = "minor"


@dataclass(frozen=True)
class IssueGroup:
    name: str
    description: str
    
class IssueLevel(str, Enum):
    MAJOR = "major"
    MEDIUM = "medium"
    MINOR = "minor"
    
Non_convergence = IssueGroup(
    name = 'Non convergence',
    description= 'Iterative Newton method did not converge for some input loads'
)


Line_overload  = IssueGroup(
    name = 'Line overload',
    description= 'Some lines are overloaded'
)


Trans_overload = IssueGroup(
    name = 'Transformer overload',
    description= 'Some transformers are overloaded'
)

Voltage_viol = IssueGroup(
    name = 'Volatge Violations',
    description= 'Some buses have too low or too high voltage'
)

    
def test_ieeebus(model_giskard, dataset_gisrkard, model, df):
    model.predict(df)
    results = model.results
    Issues = []
    
    not_converged_input_loads = []
    for i in range(len(results)):
     if results[i] is None:
        not_converged_input_loads.append(i)
    if len(not_converged_input_loads) > 0:
        examples_df = pd.DataFrame({'input_loads': df.iloc[not_converged_input_loads].values.reshape(-1)})
        Issues.append(
            Issue(
            model_giskard, 
            dataset_gisrkard,
            Non_convergence,
            IssueLevel.MAJOR,
            description = f'Newton-raphson method did not converge.',
            examples = examples_df)
        )
        
    line_overloads = []
    for i in range(len(results)):
        if results[i] is not None and sum(results[i]['over_line'] > 100):
            line_overloads.append(i)
    
    if len(line_overloads) > 0:
        examples_df = pd.DataFrame(
            {
                'input_loads': df.iloc[line_overloads].values.reshape(-1),
                'overloaded_lines': [model.model.line.index[results[i]['over_line'] > 100].tolist() for i in line_overloads]
            }
        )
        Issues.append(
            Issue(
            model_giskard, 
            dataset_gisrkard,
            Line_overload,
            IssueLevel.MAJOR,
            description = f'Some lines are overloaded.',
            examples = examples_df)
        )
    
    trans_overloads = []
    for i in range(len(results)):
        if results[i] is not None and sum(results[i]['over_trans'] > 100):
            trans_overloads.append(i)
    
    if len(line_overloads) > 0:
        examples_df = pd.DataFrame(
            {
                'input_loads': df.iloc[trans_overloads].values.reshape(-1),
                'overloaded_transformers': [model.model.trafo.index[results[i]['over_trans'] > 100].tolist() for i in trans_overloads]
            }
        )
        Issues.append(
            Issue(
            model_giskard, 
            dataset_gisrkard,
            Trans_overload,
            IssueLevel.MAJOR,
            description = f'Some transformers are overloaded.',
            examples = examples_df)
        )
    
    voltage_viloations = []
    max_vm = np.array(model.model.bus["max_vm_pu"].tolist())
    min_vm = np.array(model.model.bus["min_vm_pu"].tolist())
    
    for i in range(len(results)):
        if results[i] is not None and (sum(results[i]['vm'] > max_vm) or sum(results[i]['vm'] < min_vm)):
            voltage_viloations.append(i)
    
    if len(line_overloads) > 0:
        examples_df = pd.DataFrame(
            {
                'input_loads': df.iloc[voltage_viloations].values.reshape(-1),
                'buses_violated_voltage': [model.model.bus.index[(results[i]['vm'] > max_vm) | (results[i]['vm'] < min_vm)].tolist() for i in voltage_viloations]
            }
        )
        Issues.append(
            Issue(
            model_giskard, 
            dataset_gisrkard,
            Voltage_viol,
            IssueLevel.MAJOR,
            description = f'Some buses have too low or too high voltage.',
            examples = examples_df)
        )


    return Issues
    
    
    