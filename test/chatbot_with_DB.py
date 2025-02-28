import streamlit as st
import sqlite3
from src.ClsChatBot import ChatRobotOpenRouter

# Initialize OpenAI client
chatbot = ChatRobotOpenRouter()
chatbot.setup_env("./key.txt", "./config.json")
client = chatbot.initial_llm()

# Database connection context manager
class DatabaseConnection:
    def __init__(self, db_name):
        self.db_name = db_name

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.commit()
            self.conn.close()

# Initialize database
def init_database():
    with DatabaseConnection('chat_history.db') as c:
        # Add indexes for better performance
        c.execute('''CREATE TABLE IF NOT EXISTS topics
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    name TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chats
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    topic_id INTEGER, 
                    role TEXT, 
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(topic_id) REFERENCES topics(id) ON DELETE CASCADE)''')
        # Add indexes
        c.execute('''CREATE INDEX IF NOT EXISTS idx_topic_id ON chats(topic_id)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_created_at ON chats(created_at)''')

# Initialize session state
def init_session_state():
    if "current_topic_id" not in st.session_state:
        st.session_state.current_topic_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "new_topic" not in st.session_state:
        st.session_state.new_topic = ""

def create_topic():
    if st.session_state.new_topic:
        try:
            with DatabaseConnection('chat_history.db') as c:
                c.execute("INSERT INTO topics (name) VALUES (?)", (st.session_state.new_topic,))
                topic_id = c.lastrowid
            st.session_state.current_topic_id = topic_id
            st.session_state.messages = []
            # Clear the input
            st.session_state.new_topic = ""
        except sqlite3.IntegrityError:
            st.sidebar.error("Topic name already exists!")
        except Exception as e:
            st.sidebar.error(f"Error creating topic: {str(e)}")

# Get chat completion from OpenAI
def get_chat_response(messages):
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=messages,
            temperature=0.7,
            stream=True
        )
        
        full_response = ""
        message_placeholder = st.empty()
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"Error getting response from OpenAI: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Chat Assistant", layout="wide")
    
    # Initialize database and session state
    init_database()
    init_session_state()
    
    # Sidebar for topic management
    with st.sidebar:
        st.header("Topics")
        
        # New topic input with callback
        st.text_input(
            "New Topic",
            placeholder="Enter topic name",
            key="new_topic",
            on_change=create_topic
        )
        
        # Display topics
        with DatabaseConnection('chat_history.db') as c:
            c.execute("SELECT id, name FROM topics ORDER BY created_at DESC")
            topics = c.fetchall()
        
        if topics:
            st.divider()
            for topic_id, topic_name in topics:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"📝 {topic_name}", key=f"topic_{topic_id}"):
                        st.session_state.current_topic_id = topic_id
                        # Load messages for this topic
                        with DatabaseConnection('chat_history.db') as c:
                            c.execute("SELECT role, message FROM chats WHERE topic_id = ? ORDER BY created_at", (topic_id,))
                            st.session_state.messages = c.fetchall()
                with col2:
                    if st.button("🗑️", key=f"delete_{topic_id}"):
                        with DatabaseConnection('chat_history.db') as c:
                            c.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
                        if st.session_state.current_topic_id == topic_id:
                            st.session_state.current_topic_id = None
                            st.session_state.messages = []
                        st.rerun()
    
    # Main chat area
    if st.session_state.current_topic_id:
        # Get topic name
        with DatabaseConnection('chat_history.db') as c:
            c.execute("SELECT name FROM topics WHERE id = ?", (st.session_state.current_topic_id,))
            topic_name = c.fetchone()[0]
        
        st.header(f"Topic: {topic_name}")
        
        # Display messages
        for role, message in st.session_state.messages:
            with st.chat_message(role):
                st.write(message)
        
        # Chat input
        if prompt := st.chat_input("Type your message..."):
            # Add user message to state and display
            st.session_state.messages.append(("user", prompt))
            with st.chat_message("user"):
                st.write(prompt)
            
            # Save user message to database
            with DatabaseConnection('chat_history.db') as c:
                c.execute(
                    "INSERT INTO chats (topic_id, role, message) VALUES (?, ?, ?)",
                    (st.session_state.current_topic_id, "user", prompt)
                )
            
            # Get and display AI response
            with st.chat_message("assistant"):
                messages = [{"role": m[0], "content": m[1]} for m in st.session_state.messages]
                response = get_chat_response(messages)
                
                if response:
                    # Add AI response to state
                    st.session_state.messages.append(("assistant", response))
                    
                    # Save AI response to database
                    with DatabaseConnection('chat_history.db') as c:
                        c.execute(
                            "INSERT INTO chats (topic_id, role, message) VALUES (?, ?, ?)",
                            (st.session_state.current_topic_id, "assistant", response)
                        )
    else:
        st.info("👈 Please select or create a topic from the sidebar to start chatting.")

if __name__ == "__main__":
    main()
