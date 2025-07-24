from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import requests

app = Flask(__name__)

# Perplexity API Config
PERPLEXITY_API_KEY = "YOUR_API_KEY(ANY_AI_MODEL_API_KEY)"  # Replace with your key
API_URL = "https://api.perplexity.ai/chat/completions"

# Global variable to store dataframe
df_global = None


# Helper Functions
def extract_experience(work_experiences):
    return len(work_experiences) if isinstance(work_experiences, list) else 0

def infer_role(skills):
    if not skills:
        return "General"
    s = [skill.lower() for skill in skills]
    if any(k in s for k in ["react", "angular", "javascript", "frontend"]):
        return "Frontend Developer"
    elif any(k in s for k in ["python", "django", "flask", "backend"]):
        return "Backend Developer"
    elif any(k in s for k in ["machine learning", "ml", "data science"]):
        return "ML Engineer"
    elif any(k in s for k in ["aws", "devops", "kubernetes", "docker"]):
        return "DevOps Engineer"
    elif any(k in s for k in ["sql", "spark", "etl"]):
        return "Data Engineer"
    else:
        return "Software Engineer"

def score_candidate(row):
    score = 0
    skills = row["skills"] if isinstance(row["skills"], list) else []
    weight_map = {"Python": 10, "Machine Learning": 10, "React": 8, "AWS": 6}
    for skill in skills:
        score += weight_map.get(skill, 2)
    score += row["experience_years"] * 2
    return score

def get_ai_reasoning(candidates):
    candidate_text = "\n".join([
        f"{c['name']} - Role: {c['role']}, Skills: {', '.join(c['skills'])}, Exp: {c['experience_years']} yrs"
        for c in candidates
    ])
    prompt = f"""
Generate a detailed team selection report in Markdown format.
The structure must include:
- ## Team Selection: List the 5 selected candidates with roles.
- ## Candidate Analysis: Provide a Markdown table with columns:
  | Name | Role | Key Skills | Experience | Why Selected |
- ## Rationale: Explain team diversity, strengths, and why others were not selected.
- ## Potential Team Roles: Assign roles for each member.
- ## Conclusion: Summarize why this team is optimal.

Candidates:
{candidate_text}
"""
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are an expert recruiter."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No reasoning generated")
    else:
        return f"API Error: {response.status_code} - {response.text}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global df_global
    file = request.files['file']
    data = json.load(file)
    df = pd.DataFrame(data)

    df["experience_years"] = df["work_experiences"].apply(extract_experience)
    df["role"] = df["skills"].apply(infer_role)
    df["score"] = df.apply(score_candidate, axis=1)

    df_global = df

    return jsonify({
        "roles": sorted(df["role"].unique().tolist()),
        "locations": sorted(df["location"].dropna().unique().tolist()),
        "skills": sorted({s for sub in df["skills"] for s in (sub if isinstance(sub, list) else [])}),
        "candidates": df.to_dict(orient="records")
    })


@app.route('/filter', methods=['POST'])
def filter_candidates():
    global df_global
    data = request.json
    roles = data.get("roles", [])
    locations = data.get("locations", [])
    skills = data.get("skills", [])

    filtered_df = df_global.copy()
    if roles:
        filtered_df = filtered_df[filtered_df["role"].isin(roles)]
    if locations:
        filtered_df = filtered_df[filtered_df["location"].isin(locations)]
    if skills:
        filtered_df = filtered_df[filtered_df["skills"].apply(lambda s: any(skill in s for skill in skills))]

    top_candidates = filtered_df.sort_values(by="score", ascending=False).head(10)

    return jsonify({
        "filtered": filtered_df.to_dict(orient="records"),
        "top": top_candidates.to_dict(orient="records")
    })


@app.route('/final-team', methods=['POST'])
def final_team():
    selected = request.json.get("selected", [])
    final_df = df_global[df_global["name"].isin(selected)]
    reasoning = get_ai_reasoning(final_df.to_dict(orient="records"))

    return jsonify({
        "final": final_df.to_dict(orient="records"),
        "reasoning": reasoning
    })


if __name__ == '__main__':
    app.run(debug=True)
