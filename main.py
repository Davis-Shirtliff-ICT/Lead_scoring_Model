from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import openai

openai.api_key = "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"

app = FastAPI()

df_products = pd.read_csv("products.csv")

class LeadData(BaseModel):
    quote_amount: float
    quote_age_days: int
    customer_type: str
    source_channel: str
    product_type: str
    past_interactions: str
    prior_orders: int
    budget_fit: str  # "High", "Medium", "Low"
    decision_authority: str  # "Yes", "No"
    clear_need: str  # "Yes", "Unclear", "No"
    purchase_timeline: str  # "Immediate", "This Quarter", "Later"

 
def get_lead_priority(data: LeadData) -> str:
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

def calculate_bant_score(data: LeadData) -> int:
    budget_score = {"High": 30, "Medium": 20, "Low": 10}.get(data.budget_fit, 0)
    authority_score = 20 if data.decision_authority == "Yes" else 0
    need_score = {"Yes": 30, "Unclear": 15, "No": 0}.get(data.clear_need, 0)
    timeline_score = {"Immediate": 20, "This Quarter": 10, "Later": 5}.get(data.purchase_timeline, 0)
    return budget_score + authority_score + need_score + timeline_score

def get_relevant_products(product_type: str, top_n: int = 5):
    mask = df_products['category'].str.contains(product_type, case=False, na=False) | \
           df_products['tags'].str.contains(product_type, case=False, na=False)
    filtered = df_products[mask]
    if filtered.empty:
        filtered = df_products
    sample = filtered.sample(n=min(top_n, len(filtered)), random_state=42)
    return sample[['name', 'category', 'tags', 'price']].to_dict(orient="records")

def get_recommendation(data: LeadData, priority: str, bant_score: int, products: list) -> str:
    product_text = "\n".join(
        [f"- {p['name']} ({p['category']}): KES {p['price']} — Tags: {p['tags']}" for p in products]
    )
    prompt = (
        f"A sales engineer is engaging a lead with this profile:\n"
        f"- Quote Amount: {data.quote_amount}\n"
        f"- Quote Age: {data.quote_age_days} days\n"
        f"- Customer Type: {data.customer_type}\n"
        f"- Source Channel: {data.source_channel}\n"
        f"- Product Type: {data.product_type}\n"
        f"- Past Interactions: {data.past_interactions}\n"
        f"- Prior Orders: {data.prior_orders}\n"
        f"- BANT Score: {bant_score}/100\n"
        f"Lead Priority: {priority}\n\n"
        f"Recommended Products:\n{product_text}\n\n"
        f"As an AI sales assistant, suggest a detailed plan to convert this lead. Mention how to use the products above in your plan."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI sales assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating recommendation: {e}"
 
@app.post("/predict/")
def predict(data: LeadData):
    priority = get_lead_priority(data)
    bant_score = calculate_bant_score(data)
    products = get_relevant_products(data.product_type)
    recommendation = get_recommendation(data, priority, bant_score, products)
    return {
        "lead_priority": priority,
        "bant_score": bant_score,
        "recommended_products": products,
        "recommendation": recommendation
    }
