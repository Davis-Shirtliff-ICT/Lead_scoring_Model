from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from openai import OpenAI

app = FastAPI()
 
model = joblib.load("Gradient_boosting.pkl")

client = OpenAI(api_key="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")

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

    total = sum([
        quote_amount_score, quote_age_score, customer_type_score,
        source_channel_score, product_type_score,
        interactions_score, prior_orders_score
    ])

    if total >= 18:
        return "High Priority"
    elif total >= 12:
        return "Medium Priority"
    else:
        return "Low Priority"

def get_openai_recommendation(data: LeadData, priority: str):
    prompt = (
        f"A customer with this profile:\n"
        f"- Quote Amount: {data.quote_amount}\n"
        f"- Quote Age: {data.quote_age_days} days\n"
        f"- Customer Type: {data.customer_type}\n"
        f"- Source Channel: {data.source_channel}\n"
        f"- Product Type: {data.product_type}\n"
        f"- Past Interactions: {data.past_interactions}\n"
        f"- Prior Orders: {data.prior_orders}\n"
        f"Lead priority is {priority}.\n\n"
        f"What is the best follow-up action a salesperson should take to convert this lead?"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )

    return response.choices[0].message.content.strip()

@app.post("/predict/")
def predict_priority(data: LeadData):
    try:
        priority = get_lead_priority(data)
        recommendation = 'The current quota, is exceeded, check for a different API KEY!!'
        return {
            "lead_priority": priority,
            "recommendation": recommendation
        }
    except Exception as e:
        return {"error": str(e)}
