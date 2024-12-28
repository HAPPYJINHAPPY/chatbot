import streamlit as st
from volcenginesdkarkruntime import Ark

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
API_KEY = st.text_input("API Key", type="password")
if not API_KEY:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an client.
    client = Ark(api_key=API_KEY)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    # 输入框：用户输入聊天内容

    if prompt := st.chat_input("请输入您的问题:"):
        # 保存用户问题到会话状态
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

            # 调用 Ark API 获取回答

        def call_ark_api(messages):
            try:
                # 准备上下文消息
                ark_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

                # 调用方舟 API
                completion = client.chat.completions.create(
                    model="ep-20241226165134-6lpqj",
                    messages=ark_messages,
                    stream=True  # 流式响应
                )

                # 逐步收集并返回生成的内容
                response = ""
                for chunk in completion:
                    delta_content = (
                        chunk.choices[0].delta.content
                        if hasattr(chunk.choices[0].delta, "content")
                        else ""
                    )
                    yield delta_content
            except Exception as e:
                st.error(f"调用 Ark API 时出错：{e}")
                yield f"Error: {e}"

            # 创建占位符来动态显示助手的回答

        response_placeholder = st.empty()  # 创建占位符
        response = ""  # 初始化完整响应
        for partial_response in call_ark_api(st.session_state.messages):
            response += partial_response
            response_placeholder.markdown(response)  # 更新占位符内容

        # 将助手的完整回答保存到会话状态
        st.session_state.messages.append({"role": "assistant", "content": response})
