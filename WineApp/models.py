from django.db import models
import pickle
import pandas as pd
# Create your models here.
def predict(custom_dict):
    with open('ML/wine_model_bundle.pkl', 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    num_imputer = bundle['num_imputer']
    scaler = bundle['scaler']
    encoder = bundle['encoder']
    num_cols = bundle['num_cols']
    cat_cols = bundle['cat_cols']
    custom_input = pd.DataFrame([custom_dict])

    # Match types and process input
    for col in cat_cols:
        custom_input[col] = custom_input[col].astype(str)
    custom_input[cat_cols] = custom_input[cat_cols].fillna('Unknown')
    custom_input[num_cols] = num_imputer.transform(custom_input[num_cols])
    custom_input[num_cols] = scaler.transform(custom_input[num_cols])

    custom_encoded = encoder.transform(custom_input[cat_cols])
    custom_encoded_df = pd.DataFrame(custom_encoded, columns=encoder.get_feature_names_out(cat_cols))

    custom_final = pd.concat([
        pd.DataFrame(custom_input[num_cols], columns=num_cols),
        custom_encoded_df
    ], axis=1)

    # Prediction
    predicted_price = model.predict(custom_final)
    
    return predicted_price