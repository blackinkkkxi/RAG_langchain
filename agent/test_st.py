from langchain.agents import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st




st.set_page_config(page_title="Talk with your data!", page_icon="random")
st.title(":+1: Talk with your data!")

# the viewer should provide own OpeanAI API key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


# widget to upload a file
uploaded_file = st.file_uploader(
    "Upload a csv file, only with comma delimiter",
    type= "csv",
    help="Various File formats are Support"
)

if not uploaded_file:
    st.warning(
        "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
    )


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# write every message to the session state
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# now, the main component
if query := st.chat_input(placeholder="What is this data about?"):
    
    #### when the viewer asks a question in the "chat_input" widget, the question
    #### goes into the session_state with the role "user" and is being written 
    #### in the screen
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    #### create the LLM based on OpenAI's ChatGPT model
    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4", 
        openai_api_key=openai_api_key, 
        streaming=True
    )

    #### create an agent that connects the LLM with the dataset ("df") and 
    #### utilizes functionalities from OpenAI
    csv_agent = create_csv_agent(
        llm, uploaded_file, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    #### create the response, add it to the session_state and display it
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = csv_agent.run(query)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
