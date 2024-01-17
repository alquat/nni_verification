import torch
import pandas as pd

def test_model_with_custom_inputs(inflow, height, X_scaler, y_scaler, model):
    custom_inputs =  pd.DataFrame({"inflow": [inflow], 
                  "height": [height]})
    custom_inputs_scaled = X_scaler.transform(custom_inputs)
    custom_inputs_scaled_tensor = torch.tensor(custom_inputs_scaled, dtype=torch.float32)
    with torch.no_grad():
        output = model(custom_inputs_scaled_tensor)
        output_unscaled = y_scaler.inverse_transform(output)
        return custom_inputs_scaled_tensor , output_unscaled
    

def create_bounds_dataframe(lb, ub, y_hat, inflow, height, y_scaler):
    
    lb = lb.detach().cpu().numpy()
    ub = ub.detach().cpu().numpy()
    lb = y_scaler.inverse_transform(lb)
    ub = y_scaler.inverse_transform(ub)
    dataset = []
    pump_labels = ["pump1", "pump2"]

    for j in range(2):
        lower_bound = lb[0][j]
        upper_bound = ub[0][j]
        row = {
            "inflow": inflow,
            "height": height,
            "pump label": pump_labels[j],
            "pump_rpm estimated": y_hat[0][j],
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        dataset.append(row)

    return pd.DataFrame(dataset)