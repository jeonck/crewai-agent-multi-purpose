import streamlit as st
import os
import time
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
import dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables
dotenv.load_dotenv()

st.title("CrewAI Multi-Purpose Agent Configuration")

# Azure OpenAI configurations
deployment_name = st.text_input("Deployment Name", os.environ.get('CHAT_MODEL', ''))
api_key = st.text_input("API Key", os.environ.get("AZURE_OPENAI_API_KEY", ''), type="password")
azure_endpoint = st.text_input("Azure Endpoint", os.environ.get('AZURE_OPENAI_ENDPOINT', ''))
api_version = st.text_input("API Version", os.environ.get('OPENAI_API_VERSION', ''))

# Initialize Azure LLM
azure_llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    deployment_name=deployment_name
)

# Set up tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
web_rag_tool = WebsiteSearchTool()

# Create Agents
st.subheader("Agent Configuration")

# Researcher Agent settings
st.markdown("### Researcher Agent")
researcher_role = st.text_input("Role", "시니어 컨설턴트 겸 리서처")
researcher_goal = st.text_input("Goal", "웹에서 검색해서 분석 후 리서치 정리")
researcher_backstory = st.text_area("Backstory", """
다수의 정보를 활용하여 항상 최고의 리서치 결과를 만드는 시니어 리서처. 출처가 사실인지 체크하고 구체적인 예와 통찰을 포함.
""")

researcher = Agent(
    role=researcher_role,
    goal=researcher_goal,
    backstory=researcher_backstory,
    llm=azure_llm,
    tools=[search_tool, scrape_tool, web_rag_tool]
)

# Blog Editor Agent settings
st.markdown("### Blog Editor Agent")
editor_role = st.text_input("Editor Role", "전문적인 블로거/에디터")
editor_goal = st.text_input("Editor Goal", "블로그 글 작성")
editor_backstory = st.text_area("Editor Backstory", """
읽기 쉽고 유익한 콘텐츠를 작성하는 IT/개발 분야의 파워 블로거. 쉬운 설명, 샘플 코드를 바탕으로 구체적인 예시와 통찰을 포함한 블로그 글 작성.
""")

editor = Agent(
    role=editor_role,
    goal=editor_goal,
    backstory=editor_backstory,
    llm=azure_llm,
    verbose=True
)

# Define Tasks
st.subheader("Task Configuration")

research_task_description = st.text_input("Research Task Description", "웹에서 최신 AI 트렌드 정보 검색")
research_expected_output = st.text_input("Research Expected Output", "AI 트렌드에 대한 상세 리포트")

research_task = Task(
    description=research_task_description,
    agent=researcher,
    expected_output=research_expected_output
)

writing_task_description = st.text_input("Writing Task Description", "검색된 정보를 바탕으로 블로그 글 작성")
writing_expected_output = st.text_input("Writing Expected Output", "도입부, 주요 내용, 구체적인 예시와 출처 포함한 블로그 글")

writing_task = Task(
    description=writing_task_description,
    agent=editor,
    expected_output=writing_expected_output
)

# Corrected Crew Configuration & Execution
st.subheader("Crew Configuration & Execution")
process_type = st.selectbox("Process Type", options=[Process.sequential], format_func=lambda x: x.name)

if st.button("Run Crew"):
    with st.spinner('Running the crew...'):
        start_time = time.time()
        crew = Crew(
            agents=[researcher, editor],
            tasks=[research_task, writing_task],
            process=process_type,
            verbose=2
        )
        
        # Run the tasks
        result = crew.kickoff(inputs=dict(topic="The AI Agent Platform CrewAI"))
        end_time = time.time()
        elapsed_time = end_time - start_time
        
    st.write("Tasks completed!")
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
    st.write(result)
