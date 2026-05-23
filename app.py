import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any

from backend.ranking import CandidateRankingEngine
from backend.scorer import PERSONA_WEIGHTS

# 1. Page Configuration
st.set_page_config(
    page_title="AI Recruiter Copilot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Sleek CSS Styles
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #1E293B;
    }
    
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic card styling */
    .candidate-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E2E8F0;
        transition: all 0.3s ease;
        margin-bottom: 16px;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        border-color: #CBD5E1;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3730A3 0%, #1D4ED8 100%);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* Alert cards */
    .fraud-alert {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .clean-alert {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    
    /* Badge metrics */
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-primary { background-color: #EEF2FF; color: #4F46E5; }
    .badge-success { background-color: #DCFCE7; color: #15803D; }
    .badge-danger { background-color: #FEE2E2; color: #991B1B; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration
st.sidebar.markdown("<h2 style='text-align: center; color: #4F46E5; font-weight: 800; font-family: Outfit;'>Recruiter Copilot</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# API Setup
st.sidebar.subheader("🔌 API Integrations")
llm_provider = st.sidebar.selectbox("LLM Provider", ["Mock (Offline Demo Mode)", "OpenAI", "Gemini"])
api_key = ""
if llm_provider != "Mock (Offline Demo Mode)":
    api_key = st.sidebar.text_input("Enter API Key", type="password", help=f"Your {llm_provider} API key")
    if not api_key:
        st.sidebar.info("Using system env keys if available, otherwise defaulting to Mock mode.")

# Weights configuration / Recruiter Personas
st.sidebar.subheader("🎯 Recruiter Persona Presets")
persona_choice = st.sidebar.selectbox(
    "Select Persona",
    ["Standard / Balanced", "Tech Lead Focus (Code & Projects)", "Culture & Growth Focus (Potential & Soft Skills)", "Custom Weights"]
)

# Set weights based on persona
weights = {}
if persona_choice == "Standard / Balanced":
    weights = PERSONA_WEIGHTS["standard"]
elif persona_choice == "Tech Lead Focus (Code & Projects)":
    weights = PERSONA_WEIGHTS["tech_lead"]
elif persona_choice == "Culture & Growth Focus (Potential & Soft Skills)":
    weights = PERSONA_WEIGHTS["culture_growth"]
else:
    # Custom weights sliders
    st.sidebar.markdown("**Fine-Tune Weights (Must sum to 100%)**")
    w_sem = st.sidebar.slider("Semantic Match", 0, 100, 35)
    w_skl = st.sidebar.slider("Skill Alignment", 0, 100, 20)
    w_exp = st.sidebar.slider("Experience Relevance", 0, 100, 15)
    w_prj = st.sidebar.slider("Project Quality", 0, 100, 10)
    w_beh = st.sidebar.slider("Behavioral Signals", 0, 100, 10)
    w_gro = st.sidebar.slider("Growth Potential", 0, 100, 10)
    
    # Normalize weights to sum to 1.0
    total = w_sem + w_skl + w_exp + w_prj + w_beh + w_gro
    if total > 0:
        weights = {
            "semantic": w_sem / total,
            "skills": w_skl / total,
            "experience": w_exp / total,
            "projects": w_prj / total,
            "behavioral": w_beh / total,
            "growth": w_gro / total
        }
    else:
        weights = PERSONA_WEIGHTS["standard"]

# Show weights preview in sidebar
st.sidebar.markdown("**Current Weight Weights:**")
for k, v in weights.items():
    st.sidebar.markdown(f"- *{k.title()}*: {v*100:.1f}%")

st.sidebar.subheader("⚙️ Settings")
anonymous_mode = st.sidebar.checkbox("Anonymous Recruiting Mode", value=False, help="Hides names and contact details to reduce hiring bias")
show_fraud_alerts = st.sidebar.checkbox("Show Fraud / Copy-Paste Flags", value=True, help="Highlights candidates who copy JD phrasing verbatim")

# 4. Main Panel UI
st.markdown("<h1 class='main-title'>AI Recruiter Copilot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Production-grade candidate parsing, scoring, ranking, and explanation engine.</p>", unsafe_allow_html=True)

# Helper functions for sample data loading
def load_sample_jd():
    try:
        with open(os.path.join("data", "sample_jds", "backend_engineer.txt"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Position: Backend Engineer\nSkills: Python, FastAPI, SQL, Docker, AWS"

def load_sample_resumes_paths() -> List[str]:
    resumes_dir = os.path.join("data", "sample_resumes")
    if os.path.exists(resumes_dir):
        files = [f for f in os.listdir(resumes_dir) if f.endswith(".txt") or f.endswith(".pdf")]
        return [os.path.join(resumes_dir, f) for f in files]
    return []

# File Input Forms
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 1. Job Description")
    sample_jd_btn = st.button("🔌 Load Sample JD (Senior Backend Engineer)")
    
    jd_default_val = ""
    if sample_jd_btn:
        jd_default_val = load_sample_jd()
        
    jd_input = st.text_area("Paste Job Description here:", value=jd_default_val, height=300, placeholder="Enter job summary, required skills, preferred qualifications...")

with col2:
    st.markdown("### 👥 2. Candidates / Resumes")
    upload_method = st.radio("Upload Source", ["Multiple Resume Files (PDF/TXT)", "Structured CSV Batch", "Quick Load Samples"])
    
    uploaded_files = []
    csv_file = None
    
    if upload_method == "Multiple Resume Files (PDF/TXT)":
        uploaded_files = st.file_uploader("Upload PDF or TXT Resumes", accept_multiple_files=True, type=["pdf", "txt"])
    elif upload_method == "Structured CSV Batch":
        csv_file = st.file_uploader("Upload Batch CSV", type=["csv"], help="Must contain Name, Email, and Resume_Text or skills columns")
    else:
        sample_paths = load_sample_resumes_paths()
        st.info(f"Ready to load {len(sample_paths)} sample resumes from the data directory.")
        for path in sample_paths:
            st.markdown(f"- `{os.path.basename(path)}`")

# 5. Process and Run Ranking Engine
run_btn = st.button("🚀 Run Copilot Intelligence", use_container_width=True)

if run_btn:
    if not jd_input.strip():
        st.error("Please enter a Job Description.")
    elif upload_method == "Multiple Resume Files (PDF/TXT)" and not uploaded_files:
        st.error("Please upload one or more resumes.")
    elif upload_method == "Structured CSV Batch" and not csv_file:
        st.error("Please upload a CSV candidates dataset.")
    else:
        with st.spinner("Processing documents, running semantic analysis, and evaluating fit..."):
            # Initialize engine
            prov = "mock" if "Mock" in llm_provider else llm_provider.lower()
            engine = CandidateRankingEngine(provider=prov, api_key=api_key, weights=weights)
            
            ranked_candidates = []
            
            # Case 1: Multiple Uploaded Files
            if upload_method == "Multiple Resume Files (PDF/TXT)":
                # Create temporary directory inside scratch to store uploaded files
                temp_dir = os.path.join("scratch", "temp_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                
                temp_paths = []
                for f in uploaded_files:
                    path = os.path.join(temp_dir, f.name)
                    with open(path, "wb") as temp_f:
                        temp_f.write(f.getbuffer())
                    temp_paths.append(path)
                
                ranked_candidates = engine.evaluate_candidates(jd_input, temp_paths)
                
                # Cleanup files
                for p in temp_paths:
                    try: os.remove(p)
                    except: pass
                    
            # Case 2: CSV Batch Upload
            elif upload_method == "Structured CSV Batch":
                temp_csv = os.path.join("scratch", "temp_batch.csv")
                with open(temp_csv, "wb") as temp_f:
                    temp_f.write(csv_file.getbuffer())
                
                ranked_candidates = engine.evaluate_csv_dataset(jd_input, temp_csv)
                
                try: os.remove(temp_csv)
                except: pass
                
            # Case 3: Quick Load Samples
            else:
                sample_paths = load_sample_resumes_paths()
                if not sample_paths:
                    st.error("No sample resumes found in `data/sample_resumes`.")
                else:
                    ranked_candidates = engine.evaluate_candidates(jd_input, sample_paths)

            # Store in session state
            st.session_state["ranked_candidates"] = ranked_candidates
            st.session_state["engine"] = engine
            st.success(f"Successfully ranked {len(ranked_candidates)} candidates!")

# 6. Display Dashboard Results
if "ranked_candidates" in st.session_state and st.session_state["ranked_candidates"]:
    candidates = st.session_state["ranked_candidates"]
    engine = st.session_state["engine"]
    
    st.markdown("---")
    st.markdown("## 📊 Recruiter Intelligence Dashboard")
    
    # 6.1 Metric Summary Row
    suspicious_count = sum(1 for c in candidates if c["fraud_report"]["is_suspicious"])
    avg_score = sum(c["final_score"] for c in candidates) / len(candidates)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Evaluated", len(candidates))
    with m_col2:
        top_name = "Candidate 1"
        if candidates:
            top_name = "Anonymous " + str(candidates[0]["rank"]) if anonymous_mode else candidates[0]["name"]
        st.metric("Best Match Candidate", top_name)
    with m_col3:
        st.metric("Average Recruiter Score", f"{avg_score:.1f} / 100")
    with m_col4:
        st.metric("Flagged Fraud Risk", f"{suspicious_count} Candidates", delta=f"+{suspicious_count}" if suspicious_count > 0 else None, delta_color="inverse")

    # 6.2 View Tab Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Candidate Leaderboard", 
        "🔍 In-Depth Recruiter Evaluations", 
        "📊 Score Comparisons & Analytics", 
        "📤 Export Reports"
    ])
    
    # --- Tab 1: Candidate Leaderboard ---
    with tab1:
        st.markdown("### Match Leaderboard")
        
        # Build display table
        table_rows = []
        for c in candidates:
            c_name = f"Anonymous Candidate {c['rank']}" if anonymous_mode else c["name"]
            c_email = "Hidden" if anonymous_mode else c["email"]
            
            fraud_status = "CLEAN"
            if c["fraud_report"]["is_suspicious"]:
                fraud_status = "⚠️ SUSPICIOUS" if show_fraud_alerts else "CLEAN"
                
            row = {
                "Rank": c["rank"],
                "Candidate Name": c_name,
                "Email": c_email,
                "Final Score": c["final_score"],
                "Semantic": c["scores"]["semantic"],
                "Skills": c["scores"]["skills"],
                "Experience": c["scores"]["experience"],
                "Projects": c["scores"]["projects"],
                "Behavioral": c["scores"]["behavioral"],
                "Growth": c["scores"]["growth"],
                "Fraud Risk": fraud_status
            }
            table_rows.append(row)
            
        df_display = pd.DataFrame(table_rows)
        
        # Render a beautiful styled dataframe
        st.dataframe(
            df_display.style.background_gradient(subset=["Final Score"], cmap="Blues", vmin=40, vmax=100)
            .background_gradient(subset=["Semantic", "Skills", "Experience", "Projects", "Behavioral", "Growth"], cmap="BuGn", vmin=30, vmax=100),
            use_container_width=True,
            hide_index=True
        )
        
    # --- Tab 2: In-Depth Recruiter Evaluations ---
    with tab2:
        st.markdown("### Profile Deep Dive & Recruiter Commentary")
        
        # Select Candidate
        cand_names = [f"Rank {c['rank']}: {'Anonymous Candidate ' + str(c['rank']) if anonymous_mode else c['name']}" for c in candidates]
        selected_index_name = st.selectbox("Select Candidate to Evaluate", cand_names)
        
        # Find candidate object
        selected_idx = cand_names.index(selected_index_name)
        cand = candidates[selected_idx]
        
        # Layout: Score and commentary
        c_col1, c_col2 = st.columns([1, 1.5])
        
        with c_col1:
            st.markdown(f"#### 📊 Fit Score: **{cand['final_score']} / 100**")
            # Score radial/bar charts
            categories = ["Semantic", "Skills", "Experience", "Projects", "Behavioral", "Growth"]
            scores_list = [
                cand["scores"]["semantic"],
                cand["scores"]["skills"],
                cand["scores"]["experience"],
                cand["scores"]["projects"],
                cand["scores"]["behavioral"],
                cand["scores"]["growth"]
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=scores_list,
                y=categories,
                orientation='h',
                marker=dict(
                    color='rgba(79, 70, 229, 0.8)',
                    line=dict(color='rgb(79, 70, 229)', width=1)
                )
            ))
            fig.update_layout(
                title="Sub-score Component Breakdown",
                xaxis=dict(range=[0, 100]),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence rating
            st.markdown(f"**Recruiter Copilot Trust Confidence:** `{cand['reasoning']['confidence_score']}%`")
            st.progress(cand['reasoning']['confidence_score'] / 100.0)

        with c_col2:
            st.markdown("#### 🗣️ Recruiter Evaluation & Reasoning")
            st.info(cand["reasoning"]["fit_summary"])
            
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("<p class='metric-badge badge-success'>🟢 Key Strengths</p>", unsafe_allow_html=True)
                for s in cand["reasoning"]["strengths"]:
                    st.markdown(f"- {s}")
            with r_col2:
                st.markdown("<p class='metric-badge badge-warning'>🟡 Gaps / Weaknesses</p>", unsafe_allow_html=True)
                for w in cand["reasoning"]["weaknesses"]:
                    st.markdown(f"- {w}")
                    
            # Custom Interview Questions
            st.markdown("#### 🗣️ Recommended Interview Questions")
            for idx, q in enumerate(cand["reasoning"]["interview_questions"], 1):
                st.markdown(f"**Q{idx}:** *{q}*")

        # Fraud Alert Card
        if show_fraud_alerts and cand["fraud_report"]["is_suspicious"]:
            st.markdown(
                f"""
                <div class="fraud-alert">
                    <h5 style="color: #991B1B; margin: 0 0 8px 0;">⚠️ Fraud Detection Alert</h5>
                    <p style="color: #7F1D1D; margin: 0 0 8px 0;">{cand['fraud_report']['reason']}</p>
                    <p style="color: #7F1D1D; font-size: 0.85rem; margin: 0;"><b>Verbatim segments flagged:</b> {", ".join(cand['fraud_report']['copied_phrases'])}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Candidate Details Sections (Skills, Education, Experience, Projects)
        st.markdown("---")
        st.markdown("### 🔎 Parsed Candidate Profile Details")
        
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("**Skills:**")
            st.markdown(", ".join([f"`{s}`" for s in cand["skills"]]))
            
            st.markdown("**Tools & DevOps:**")
            st.markdown(", ".join([f"`{t}`" for t in cand["tools"]]))
            
            st.markdown("**Education:**")
            for edu in cand["education"]:
                st.markdown(f"- **{edu.get('degree')} in {edu.get('major')}** at {edu.get('school')} (Grad: {edu.get('grad_date')})")

        with p_col2:
            st.markdown("**Work Experience:**")
            for exp in cand["experience"]:
                st.markdown(f"- **{exp.get('title')}** at *{exp.get('company')}* ({exp.get('dates')}) - *{exp.get('years')} years*")
                st.markdown(f"  *{exp.get('description')}*")
                
            st.markdown("**Projects:**")
            for prj in cand["projects"]:
                st.markdown(f"- **{prj.get('title')}**: {prj.get('description')}")
                st.markdown(f"  *Techs: {', '.join(prj.get('technologies', []))}*")

    # --- Tab 3: Score Comparisons & Analytics ---
    with tab3:
        st.markdown("### Candidate Match Comparisons")
        
        # Parallel coordinates or multi bar comparison
        comp_df = pd.DataFrame([
            {
                "Candidate": f"Rank {c['rank']}: {'Anon ' + str(c['rank']) if anonymous_mode else c['name']}",
                "Semantic": c["scores"]["semantic"],
                "Skills": c["scores"]["skills"],
                "Experience": c["scores"]["experience"],
                "Projects": c["scores"]["projects"],
                "Behavioral": c["scores"]["behavioral"],
                "Growth": c["scores"]["growth"]
            } for c in candidates
        ])
        
        melted_df = comp_df.melt(id_vars="Candidate", var_name="Component", value_name="Score")
        
        fig_comp = px.bar(
            melted_df,
            x="Component",
            y="Score",
            color="Candidate",
            barmode="group",
            title="Component Scores Side-by-Side Comparison",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_comp.update_layout(yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- Tab 4: Export Reports ---
    with tab4:
        st.markdown("### Export Candidates Evaluations")
        
        # Export CSV Dataframe
        export_df = engine.export_to_dataframe(candidates)
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Ranked Candidates (CSV)",
            data=csv_data,
            file_name="ranked_candidates.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Export JSON evaluations
        json_data = json.dumps(candidates, indent=2).encode('utf-8')
        st.download_button(
            label="📥 Download Full Recruiter Reports (JSON)",
            data=json_data,
            file_name="candidate_evaluations.json",
            mime="application/json",
            use_container_width=True
        )
else:
    # Landing page helper instructions
    st.markdown("---")
    st.markdown("### How to Use the AI Recruiter Copilot:")
    col_inf1, col_inf2, col_inf3 = st.columns(3)
    with col_inf1:
        st.markdown("##### 1. Input Job Description")
        st.write("Either paste a custom Job Description or click the **Load Sample JD** button to load a pre-configured Senior Backend role JD.")
    with col_inf2:
        st.markdown("##### 2. Upload Candidate Resumes")
        st.write("Upload multiple candidate resumes (PDF or TXT formats) or a CSV batch. Alternatively, select **Quick Load Samples** to use mock resumes from the data directory.")
    with col_inf3:
        st.markdown("##### 3. Run and Analyze")
        st.write("Click **Run Copilot Intelligence** to evaluate candidates. You'll get score breakdowns, strengths/weaknesses reviews, custom interview questions, and fraud detection.")
