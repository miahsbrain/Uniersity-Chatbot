import streamlit as st
from model import chat


# Create storage
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
# st.write(st.session_state.messages)

for message in st.session_state.messages:
        with st.chat_message(name=message.get('role')):
            st.write(message.get('content'))


prompt = st.chat_input('Ask anything...')

if prompt:
    # Append to user input storage
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Display typed message
    with st.chat_message(name='user'):
        st.write(prompt)

    # Get user reply and append to session storage
    reply = str(chat(prompt))
    st.session_state.messages.append({'role': 'assistant', 'content': reply})

    # Display reply
    with st.chat_message(name='assistant'):
        st.write(reply)