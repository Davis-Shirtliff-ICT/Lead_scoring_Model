from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("Gradient_boosting.pkl")

# Define input data format
class LeadData(BaseModel):
    quote_amount: float
    quote_age_days: int
    customer_type: str
    source_channel: str
    product_type: str
    past_interactions: str
    prior_orders: int

def get_lead_priority(data: LeadData):
    def score(value, thresholds, scores):
        for t, s in zip(thresholds, scores):
            if value <= t:
                return s
        return scores[-1]

    quote_amount_score = score(data.quote_amount, [50000, 150000], [1, 2, 3])
    quote_age_score = score(data.quote_age_days, [5, 15], [3, 2, 1])
    customer_type_score = 3 if data.customer_type in ['Contractor', 'Corporate'] else 2 if data.customer_type in ['Reseller', 'Government'] else 1
    source_channel_score = 3 if data.source_channel in ['Website', 'Referral'] else 2 if data.source_channel in ['Walk-in', 'Email'] else 1
    product_type_score = 3 if data.product_type == 'High-margin' else 1 if data.product_type == 'Low-margin' else 2
    interactions_score = 3 if data.past_interactions in ['5 (calls, visits)', '6 (calls, visits, emails)'] else 2 if data.past_interactions in ['3 (calls, visits)', '2 (calls)'] else 1
    prior_orders_score = 3 if data.prior_orders >= 10 else 2 if data.prior_orders >= 5 else 1

    total = sum([quote_amount_score, quote_age_score, customer_type_score,
                 source_channel_score, product_type_score,
                 interactions_score, prior_orders_score])

    if total >= 18:
        return "High Priority"
    elif total >= 12:
        return "Medium Priority"
    else:
        return "Low Priority"

@app.post("/predict/")
def predict_priority(data: LeadData):
    priority = get_lead_priority(data)
    return {"lead_priority": priority}
