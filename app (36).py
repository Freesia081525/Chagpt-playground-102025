import os
import ast
import json
import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# ----------------------------
# App configuration
# ----------------------------
st.set_page_config(page_title="Agentic Mind Graph", page_icon="🧠", layout="wide")

# ----------------------------
# Sidebar: Model Provider + API Key
# ----------------------------
st.sidebar.subheader("Model Provider")
provider = st.sidebar.radio("Choose provider:", ["Gemini", "Grok"])

if provider == "Gemini":
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    model_name = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
    if api_key:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
else:
    api_key = st.sidebar.text_input("Enter Grok API Key", type="password")
    model_name = st.sidebar.selectbox("Grok Model", ["grok-3-mini", "grok-4-fast-reasoning"])
    if api_key:
        from xai_sdk import Client
        from xai_sdk.chat import user as grok_user
        client = Client(api_key=api_key, timeout=3600)

# ----------------------------
# Sidebar: Theme Selector
# ----------------------------
st.sidebar.subheader("🎨 Theme Selector")

themes = {
    "Sky Blue": {"bg": "#E6F7FF", "edge": "#3399FF", "font": "#003366"},
    "Snow White": {"bg": "#FFFFFF", "edge": "#999999", "font": "#000000"},
    "Alp. Forest": {"bg": "#E8F5E9", "edge": "#2E7D32", "font": "#1B5E20"},
    "Deep Sea": {"bg": "#001F3F", "edge": "#0074D9", "font": "#7FDBFF"},
    "Ferrari Sportscar": {"bg": "#FFEBEE", "edge": "#C62828", "font": "#B71C1C"},
    "Fendi Casa Luxury": {"bg": "#FFF8E1", "edge": "#6D4C41", "font": "#4E342E"}
}

theme_choice = st.sidebar.selectbox("Choose a theme", list(themes.keys()))
theme = themes[theme_choice]

# Node color picker
node_color = st.sidebar.color_picker("Pick node color", "#FF7F50")  # default coral

# ----------------------------
# Helper functions
# ----------------------------
def parse_text_records(text: str):
    try:
        df = pd.read_csv(pd.io.common.StringIO(text))
        return df
    except Exception:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return pd.DataFrame({"record": lines})

def load_dataset(file, paste_text):
    if file is not None:
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith(".json"):
            data = json.load(file)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        elif name.endswith(".txt"):
            content = file.read().decode("utf-8", errors="ignore")
            df = parse_text_records(content)
        else:
            st.error("Unsupported file format.")
            return None, None
    elif paste_text:
        try:
            data = json.loads(paste_text)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        except Exception:
            try:
                df = pd.read_csv(pd.io.common.StringIO(paste_text))
            except Exception:
                df = parse_text_records(paste_text)
    else:
        return None, None
    return df, df.to_dict(orient="records")

def heuristic_relationships(records):
    pairs = set()
    for r in records:
        project = r.get("project") or r.get("title")
        team = r.get("team") or r.get("department")
        author = r.get("author")
        log = r.get("log_entry") or r.get("log")
        if project and team: pairs.add((project, team))
        if project and author: pairs.add((project, f"Author: {author}"))
        if project and log: pairs.add((project, log[:50]))
    return list(pairs)

def prompt_relationships(records, instruction=None):
    context = "\n".join([json.dumps(r, ensure_ascii=False) for r in records[:50]])
    base_instruction = (
        "From the dataset records, extract main topics, sub-topics, and their relationships. "
        "Return ONLY a valid Python list of tuples [('Source','Target'), ...]."
    )
    if instruction:
        base_instruction += f"\nAdditional guidance: {instruction}"

    if provider == "Gemini" and api_key:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(f"Records:\n{context}\n\nTask:\n{base_instruction}")
            text = resp.text.strip()
            if text.startswith("```"): text = "\n".join(text.splitlines()[1:-1])
            return ast.literal_eval(text)
        except Exception as e:
            st.warning(f"Gemini error: {e}")
            return heuristic_relationships(records)

    elif provider == "Grok" and api_key:
        try:
            chat = client.chat.create(model=model_name)
            chat.append(grok_user(f"Records:\n{context}\n\nTask:\n{base_instruction}"))
            resp = chat.sample()
            text = resp.content.strip()
            if text.startswith("```"): text = "\n".join(text.splitlines()[1:-1])
            return ast.literal_eval(text)
        except Exception as e:
            st.warning(f"Grok error: {e}")
            return heuristic_relationships(records)

    else:
        st.warning("No API key provided. Using heuristic extraction.")
        return heuristic_relationships(records)

def build_graph_elements(relationships):
    nodes_set = set()
    for s, t in relationships: nodes_set.add(s); nodes_set.add(t)
    nodes = [Node(id=i, label=name, size=25, color=node_color) for i, name in enumerate(nodes_set)]
    node_index = {n.label: n.id for n in nodes}
    edges = [Edge(source=node_index[s], target=node_index[t], color=theme["edge"]) for s, t in relationships]
    return nodes, edges, node_index

def records_by_node(node_label, records):
    results = []
    for r in records:
        if node_label.lower() in json.dumps(r, ensure_ascii=False).lower():
            results.append(r)
    return results

# ----------------------------
# Main UI
# ----------------------------
st.title("🧠 Agentic Mind Graph (Gemini / Grok)")

uploaded = st.file_uploader("Upload CSV / JSON / TXT", type=["csv","json","txt"])
paste_text = st.text_area("Or paste dataset content", height=150)
custom_instruction = st.text_area("Optional guidance", height=100)
infer_button = st.button("Infer relationships")

df, records = load_dataset(uploaded, paste_text)
if df is None:
    st.info("Upload or paste data to begin.")
    st.stop()

st.dataframe(df.head(), use_container_width=True)

if "relationships" not in st.session_state:
    st.session_state.relationships = heuristic_relationships(records)

if infer_button:
    st.session_state.relationships = prompt_relationships(records, custom_instruction)

rels_df = pd.DataFrame(st.session_state.relationships, columns=["source","target"])
edited_df = st.data_editor(rels_df, num_rows="dynamic")
st.session_state.relationships = list(edited_df.itertuples(index=False, name=None))

nodes, edges, node_index = build_graph_elements(st.session_state.relationships)
config = Config(
    width=900,
    height=600,
    directed=True,
    physics=True,
    nodeHighlightBehavior=True,
    highlightColor=theme["edge"],
    bgcolor=theme["bg"],
    font={"color": theme["font"], "size": 12}
)

st.subheader("Interactive Mind Graph")
selected = agraph(nodes=nodes, edges=edges, config=config)

# ----------------------------
# Fixed node click handling
# ----------------------------
if selected:
    # Case 1: selected is a dict with "nodes"
    if isinstance(selected, dict) and "nodes" in selected and selected["nodes"]:
        node_id = selected["nodes"][0]
        label = [n.label for n in nodes if n.id == node_id][0]
        st.write(f"**Node:** {label}")
        st.json(records_by_node(label, records))

    # Case 2: selected is a list of node IDs
    elif isinstance(selected, list) and len(selected) > 0:
        node_id = selected[0]
        label = [n.label for n in nodes if n.id == node_id][0]
        st.write(f"**Node:** {label}")
        st.json(records_by_node(label, records))

    else:
        st.info("Click a node to view related records.")
else:
    st.info("Click a node to view related records.")
