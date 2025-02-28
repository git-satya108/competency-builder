import os
import time
import requests
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import anthropic
from tavily import TavilyClient
from youtube_search import YoutubeSearch

load_dotenv()


# =============================================================================
# Custom Lightweight Agent Framework
# =============================================================================

class Agent:
    """Base class for an agent."""

    def process(self, query: str):
        raise NotImplementedError("Each agent must implement its own process() method.")


class Memory:
    """Simple memory class to store past interactions."""

    def __init__(self):
        self.records = []

    def store(self, record: dict):
        self.records.append(record)

    def get_all(self):
        return self.records


class Team:
    """
    Orchestrates multiple agents and aggregates results.
    We fetch separate references for each level (Beginner, Intermediate, Advanced).
    """

    def __init__(self, agents: dict):
        self.agents = agents
        self.memory = Memory()

    def process_query(self, query: str, function: str, competency: str, role: str) -> dict:
        """
        1) Generate the main learning path.
        2) For each level (Beginner, Intermediate, Advanced), run separate queries
           to the web/video agents for context-specific references.
        """
        # 1. Generate initial learning path
        lp_result = self.agents["learning_path"].process(query)
        meta_prompt = (
            f"Review and critique the following learning path and suggest improvements:\n\n"
            f"{lp_result.get('content', '')}"
        )
        meta_result = self.agents["meta_reviewer"].process(meta_prompt) if lp_result else None

        final_content = (
            meta_result.get("content")
            if meta_result and len(meta_result.get("content", "")) > len(lp_result.get("content", "")) * 0.8
            else lp_result.get("content", "")
        )

        # 2. Gather references for each level
        references = {
            "beginner": {"web": [], "videos": []},
            "intermediate": {"web": [], "videos": []},
            "advanced": {"web": [], "videos": []}
        }

        def build_query(level: str) -> str:
            return f"{level} level {function} {competency} {role} training"

        for level_key in ["beginner", "intermediate", "advanced"]:
            level_query = build_query(level_key)
            references[level_key]["web"] = self.agents["web"].process(level_query)
            references[level_key]["videos"] = self.agents["video"].process(level_query)

        self.memory.store({"query": query, "learning_path": final_content})

        return {
            "content": {
                "content": final_content,
                "provider": lp_result.get("provider", "Unknown"),
                "latency": lp_result.get("latency", 0)
            },
            "references": references
        }


# =============================================================================
# Agent Implementations
# =============================================================================

class LearningPathAgent(Agent):
    """
    Generates a structured learning path with tables and a "General Suggestions" section.
    Uses both OpenAI (GPT-4 Turbo) and Anthropic (Claude) and returns the best result.
    """

    def __init__(self, openai_client, anthropic_client):
        self.llms = {"openai": openai_client, "anthropic": anthropic_client}

    def process(self, query: str):
        # We'll add instructions to produce each level in tabular format + general suggestions
        extended_prompt = (
            f"{query}\n\n"
            "Please format each level (Beginner, Intermediate, Advanced) in a Markdown table with columns:\n"
            "**Module / Course**, **URL**, and **Description**.\n"
            "After the Advanced level, include a section titled 'General Suggestions' with the following text:\n\n"
            "1. **Integration of Practical Experience**:\n"
            "   While the path includes numerous courses and certifications, integrating real-world applications "
            "   through internships, shadowing experiences, or project-based learning could enhance practical skills. "
            "   Consider recommending partnerships with HR departments in various industries to provide hands-on learning opportunities.\n"
            "2. **Soft Skills Development**:\n"
            "   HR roles require strong interpersonal skills, including communication, negotiation, and empathy. "
            "   Adding specific training focused on these soft skills, especially at the beginner and intermediate levels, could be beneficial.\n"
            "3. **Customization and Flexibility**:\n"
            "   Emphasize the importance of tailoring this path to individual needs more explicitly. "
            "   Suggest tools or methods for self-assessment to help learners identify which areas they need to focus on "
            "   based on their current skills and career aspirations.\n"
            "4. **Technology Integration**:\n"
            "   Given the rapid advancement in HR technologies, including AI and machine learning, incorporating "
            "   more advanced tech-focused courses at earlier stages could prepare learners for the digital transformation in HR.\n"
            "5. **Sustainability and Ethics**:\n"
            "   As companies increasingly prioritize sustainability and ethical practices, including training on these topics "
            "   could be valuable. This could cover ethical decision-making in HR, sustainable business practices, and "
            "   corporate social responsibility.\n\n"
            "Finally, ensure the content is well-structured and references the importance of continuous learning. "
            "End with a short concluding paragraph summarizing the holistic approach.\n\n"
            "Now produce the final output in well-structured Markdown."
        )

        versions = []
        for provider in ["openai", "anthropic"]:
            try:
                start_time = time.time()
                if provider == "anthropic":
                    response = self.llms[provider].messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4000,
                        messages=[{"role": "user", "content": extended_prompt}]
                    )
                    content = response.content[0].text
                    provider_name = "Claude-3.5-Sonnet"
                else:
                    response = self.llms["openai"].chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": extended_prompt}],
                        temperature=0.3
                    )
                    content = response.choices[0].message.content
                    provider_name = "OpenAI-GPT4"
                latency = time.time() - start_time
                versions.append({
                    "content": content,
                    "provider": provider_name,
                    "latency": latency
                })
            except Exception as e:
                st.error(f"{provider} error: {str(e)}")

        if versions:
            best = max(versions, key=lambda x: len(x["content"]))
            return best
        return {"content": "No learning path generated."}


class MetaReviewerAgent(Agent):
    """
    Acts as a meta reviewer/critic to improve the generated learning path.
    """

    def __init__(self, openai_client, anthropic_client):
        self.llms = {"openai": openai_client, "anthropic": anthropic_client}

    def process(self, query: str):
        prompt = query
        versions = []
        for provider in ["openai", "anthropic"]:
            try:
                start_time = time.time()
                if provider == "anthropic":
                    response = self.llms[provider].messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=2000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                    provider_name = "Claude-3.5-Sonnet"
                else:
                    response = self.llms["openai"].chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    content = response.choices[0].message.content
                    provider_name = "OpenAI-GPT4"
                latency = time.time() - start_time
                versions.append({
                    "content": content,
                    "provider": provider_name,
                    "latency": latency
                })
            except Exception as e:
                st.error(f"MetaReviewer {provider} error: {str(e)}")

        if versions:
            best = max(versions, key=lambda x: len(x["content"]))
            return best
        return {"content": "No review generated."}


class WebSearchAgent(Agent):
    """
    Retrieves relevant web resources using Tavily and Google Serper APIs.
    """

    def __init__(self, tavily_client, serper_config: dict):
        self.tavily = tavily_client
        self.serper_config = serper_config

    def process(self, query: str):
        results = []
        try:
            tavily_results = self.tavily.search(query=query, max_results=5)
            results.extend([
                {"title": r.get("title", "Untitled"), "url": r["url"]}
                for r in tavily_results.get("results", []) if "url" in r
            ])
            response = requests.post(
                self.serper_config["url"],
                headers=self.serper_config["headers"],
                json={"q": query, "num": 5},
                timeout=self.serper_config["timeout"]
            )
            serper_results = response.json()
            results.extend([
                {"title": r.get("title", "Untitled"), "url": r["link"]}
                for r in serper_results.get("organic", []) if "link" in r
            ])
            seen = set()
            deduped = [x for x in results if not (x["url"] in seen or seen.add(x["url"]))]
            return deduped[:10]
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []


class VideoSearchAgent(Agent):
    """
    Retrieves relevant video guides using the YouTube Search API.
    """

    def process(self, query: str):
        try:
            results = YoutubeSearch(query, max_results=10).to_dict()
            videos = [{
                "title": r.get("title", "Untitled Video"),
                "url": f"https://youtube.com/watch?v={r['id']}",
                "views": r.get("views", "N/A")
            } for r in results if 'id' in r]
            return videos[:5]
        except Exception as e:
            st.error(f"Video search error: {str(e)}")
            return []


# =============================================================================
# Streamlit Application
# =============================================================================

def main():
    st.set_page_config(
        page_title="Function → Competency → Role Learning Path",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded"
    )
    st.title("📊 Function → Competency → Role Learning Path Generator")
    st.markdown("### AI-Powered Multi-Level Training Recommendations")
    st.write("---")

    # Sidebar: Upload Excel file with columns: [Function, Competency, Role]
    st.sidebar.header("Upload Mapping File")
    uploaded_file = st.sidebar.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

    # Data structure -> { "Human Resources": { "HR Management": ["HR Director", ...], ... }, ... }
    function_dict = {}

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            # Expecting columns named exactly: "Function", "Competency", "Role"
            for func in df["Function"].unique():
                sub_df = df[df["Function"] == func]
                competency_map = {}
                for comp in sub_df["Competency"].unique():
                    roles = sub_df[sub_df["Competency"] == comp]["Role"].tolist()
                    competency_map[comp] = roles
                function_dict[func] = competency_map
            st.sidebar.success("Mapping loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

    # Step 1: Choose Function
    if function_dict:
        function_list = list(function_dict.keys())
    else:
        function_list = ["Human Resources", "Information Technology", "Finance"]
    selected_function = st.selectbox("Select Function", function_list)

    # Step 2: Choose Competency
    if selected_function in function_dict:
        competency_list = list(function_dict[selected_function].keys())
        selected_competency = st.selectbox("Select Competency", competency_list)
    else:
        selected_competency = st.text_input("Enter Competency", "HR Management")

    # Step 3: Choose Role
    role = None
    if selected_function in function_dict and selected_competency in function_dict[selected_function]:
        role_list = function_dict[selected_function][selected_competency]
        role = st.selectbox("Select Role", role_list)
    else:
        role = st.text_input("Enter Role", "HR Manager")

    # Generate learning path if button is pressed
    if st.button("Generate Learning Path"):
        if selected_function and selected_competency and role:
            # Construct prompt for multi-level learning path (table format + appended general suggestions)
            prompt = (
                f"Generate a detailed and well-structured learning path for a {role} "
                f"in the {selected_competency} competency within the {selected_function} function. "
                "Divide the learning path into three levels:\n\n"
                "**Beginner**: For those with no prior experience. Provide recommended introductory courses, "
                "training modules, and online resources with URLs.\n\n"
                "**Intermediate**: For those with 2-3 years of experience. Recommend advanced courses, specialized training, "
                "and certifications with resource links.\n\n"
                "**Advanced**: For those with more than 4 years of experience. Suggest expert-level learning, leadership development, "
                "and industry-recognized certifications with reference URLs.\n\n"
                "Each level should be presented in a Markdown table with columns: Module / Course, URL, Description.\n"
                "After the Advanced level, add a 'General Suggestions' section. Then end with a short concluding paragraph.\n"
                "Focus on clarity and a structured layout.\n"
            )

            # Prepare external API clients
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            serper_config = {
                "url": "https://google.serper.dev/search",
                "headers": {"X-API-KEY": os.getenv("SERPER_API_KEY")},
                "timeout": 10
            }

            # Instantiate agents
            lp_agent = LearningPathAgent(openai_client, anthropic_client)
            mr_agent = MetaReviewerAgent(openai_client, anthropic_client)
            ws_agent = WebSearchAgent(tavily_client, serper_config)
            vs_agent = VideoSearchAgent()

            agents = {
                "learning_path": lp_agent,
                "meta_reviewer": mr_agent,
                "web": ws_agent,
                "video": vs_agent
            }
            team = Team(agents)

            with st.spinner("Generating learning path and resources..."):
                result = team.process_query(prompt, selected_function, selected_competency, role)
                st.session_state.current_result = result
        else:
            st.warning("Please provide all required inputs.")

    # Display results if available
    if st.session_state.get("current_result"):
        result = st.session_state.current_result
        content_obj = result.get("content")
        references = result.get("references", {})

        doc_col, ref_col = st.columns([3, 1])
        with doc_col:
            with st.expander("📄 Generated Learning Path", expanded=True):
                st.markdown(content_obj.get("content", "No content generated."), unsafe_allow_html=True)
            with st.expander("⚙️ Generation Details"):
                latency = content_obj.get("latency", 0.0)
                provider = content_obj.get("provider", "Unknown")
                c1, c2 = st.columns(2)
                c1.metric("Processing Time", f"{latency:.2f}s")
                c2.metric("AI Engine", provider)
                st.caption("Powered by Claude-3.5-Sonnet and GPT-4 Turbo")

        # Right pane: curated resources for each level
        with ref_col:
            st.subheader("🔗 Curated Resources")

            # We'll display each level's resources in a separate expander
            for level_key in ["beginner", "intermediate", "advanced"]:
                level_title = level_key.capitalize()  # e.g. "Beginner"
                level_data = references.get(level_key, {"web": [], "videos": []})

                with st.expander(f"{level_title} Level"):
                    # Web references
                    st.markdown("**Web References**")
                    web_results = level_data["web"]
                    if web_results:
                        for i, item in enumerate(web_results[:5], 1):
                            st.markdown(f"{i}. [{item['title']}]({item['url']})")
                    else:
                        st.info("No web resources found")

                    # Video guides
                    st.markdown("**Video Guides**")
                    video_results = level_data["videos"]
                    if video_results:
                        for i, vid in enumerate(video_results[:3], 1):
                            st.markdown(f"{i}. ▶️ [{vid['title']}]({vid['url']})")
                            st.caption(f"Views: {vid.get('views', 'N/A')}")
                    else:
                        st.info("No video guides found")

            # For version history, if you maintain multiple runs
            with st.expander("🕰 Document Versions"):
                if 'history' in st.session_state and len(st.session_state.history) > 1:
                    for i, rev in enumerate(st.session_state.history):
                        c1, c2 = st.columns([1, 3])
                        c1.button(
                            f"v{i + 1}",
                            key=f"load_{i}",
                            on_click=lambda r=rev: st.session_state.update(current_result=r),
                            help=f"Load version {i + 1}"
                        )
                        # If you store provider info in rev["learning_path"], adapt accordingly
                        prov = rev["learning_path"].get("provider", "Unknown")
                        lat = rev["learning_path"].get("latency", 0.0)
                        c2.caption(f"{prov} | {lat:.1f}s")
                else:
                    st.info("No previous versions")


if __name__ == "__main__":
    main()
