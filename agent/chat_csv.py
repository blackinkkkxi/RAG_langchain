from langchain.agents import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st


st.set_page_config(page_title="Chat with your csv!!" , page_icon="random")
st.title(":male-student: :book: Chat with your csv!!")

uploaded_file = st.file_uploader(
      "请上传你需要分析的数据" ,
      type = "csv" ,
      help = "你需要上传的格式为csv"
)

if not uploaded_file:
    st.warning("您必须上传一个文件从而进行数据分析")
    
OpneAI_key = st.sidebar.text_input("OpenAI key" , type= "password")
    

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state['messages'] = [{"role" : "assistant" , "content" : "How can i help you?"}]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if query := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append( {"role" : "user"  , "content" : query})
    st.chat_message("user").write(query)
    
    if not OpneAI_key:
        st.info("请添加您的OPENAI key")
        st.stop
    
    
    llm = ChatOpenAI(
         temperature = 0 ,
         #model =
         openai_api_key = OpneAI_key , 
     )   
    
    csv_agent = create_csv_agent(
        llm , 
        uploaded_file,
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = csv_agent.run(query)
        st.session_state.messages.append({"role":"assistant" , "content":response})
        st.write(response)
