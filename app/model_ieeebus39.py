import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

def build_net(i_max_ka=1.2, vmin=0.95, vmax=1.05):
    net = pn.case39()
    # Voltage bounds
    net.bus["min_vm_pu"] = vmin
    net.bus["max_vm_pu"] = vmax
    # Thermal limits for lines and transformers
    net.line["max_i_ka"] = i_max_ka
    # Case 39 often has trafos; ensure they have loading limits
    net.trafo["max_loading_percent"] = 100.0 
    return net

def apply_total_load(net, total_p_mw, pf=0.95):
    """Scales load based on total P and a constant Power Factor."""
    base_p = net.load["p_mw"].values
    base_total = base_p.sum()
    k = float(total_p_mw) / float(base_total)
    
    net.load["p_mw"] = base_p * k
    # Q = P * tan(acos(pf))
    q_ratio = np.tan(np.arccos(pf))
    net.load["q_mvar"] = net.load["p_mw"] * q_ratio
    
class Model:
    def __init__(self, checkpoint_path: str):
        """
        Initialization of the model.
        """
        self.model = build_net()
        self.results = None
        
    def load_checkpoint(self):
        pass
    
    def predict(self, df):
        """
        Makes prediction based on the total active power load total_p_mw.
        Returns voltage magnitudes at all buses of shape (len(df),).
        """
        total_p_mws = df.values.flatten()
        results = []
        for total_p_mw in total_p_mws:
            apply_total_load(self.model, total_p_mw)
            
            
            try:
                pp.runpp(self.model, init="auto")
                
                over_line = self.model.res_line["loading_percent"].to_list()
                over_trafo = self.model.res_trafo["loading_percent"].to_list()
                vm = self.model.res_bus["vm_pu"].to_list()
                #np array
                results.append( # 35,11,39 = 
                    {
                        'over_line': np.array(over_line),
                        'over_trans': np.array(over_trafo),
                        'vm': np.array(vm)
                    }
                )
            
            except pp.LoadflowNotConverged:
                results.append(None)
        
        self.results = results
                
        return np.ones(len(results))
        
        
        