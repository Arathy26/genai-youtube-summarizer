import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import os
import re

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

def extract_video_id(url):
    patterns = [
        r'v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        r'embed/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id)
    transcript = " ".join([t.text for t in transcript_list])
    return transcript
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    return vector_db

def search_chunks(vector_db, question, k=4):
    results = vector_db.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

def summarize_video(chunks):
    context = "\n\n".join(chunks[:10])
    prompt = PromptTemplate(
        input_variables=["context"],
        template="""
        You are an expert content summarizer.
        Summarize this YouTube video transcript clearly.

        Transcript:
        {context}

        Give summary in this format:

        🎥 VIDEO TOPIC:
        (what is this video about in one line)

        📌 MAIN POINTS:
        (5-7 key points covered in the video)

        🔑 KEY TAKEAWAYS:
        (3-5 most important things to remember)

        💡 INTERESTING INSIGHTS:
        (any unique or interesting ideas mentioned)

        📝 OVERALL SUMMARY:
        (3-4 sentence complete summary)

        ⏱️ BEST FOR:
        (who should watch this video and why)
        """
    )
    chain = prompt | llm
    result = chain.invoke({"context": context}).content
    return result

def answer_question(vector_db, question):
    context = search_chunks(vector_db, question)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an expert assistant analyzing a YouTube video transcript.

        Transcript Content:
        {context}

        Question: {question}

        Give a clear detailed answer:

        📊 ANSWER:
        (direct answer from the video)

        🎯 CONTEXT FROM VIDEO:
        (relevant parts of video that support this answer)

        💡 ADDITIONAL INSIGHTS:
        (any related points from the video)
        """
    )
    chain = prompt | llm
    result = chain.invoke({
        "context": context,
        "question": question
    }).content
    return result

def generate_quiz(chunks):
    context = "\n\n".join(chunks[:8])
    prompt = PromptTemplate(
        input_variables=["context"],
        template="""
        You are an expert teacher.
        Based on this video transcript generate a quiz.

        Transcript:
        {context}

        Generate 5 multiple choice questions in this format:

        Q1: (question)
        A) option 1
        B) option 2
        C) option 3
        D) option 4
        Answer: (correct option)

        Continue for all 5 questions.
        """
    )
    chain = prompt | llm
    result = chain.invoke({"context": context}).content
    return result

st.set_page_config(
    page_title="AI YouTube Summarizer",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 AI YouTube Video Summarizer")
st.subheader("Paste any YouTube URL and chat with the video!")
st.divider()

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

st.markdown("### 🔗 Step 1 — Paste YouTube URL")
youtube_url = st.text_input(
    "YouTube URL:",
    placeholder="e.g. https://www.youtube.com/watch?v=xxxxxxxx"
)

if st.button("📥 Load Video", use_container_width=True):
    if not youtube_url.strip():
        st.warning("⚠️ Please paste a YouTube URL!")
    else:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL!")
        else:
            try:
                with st.spinner("📥 Extracting transcript..."):
                    transcript = get_transcript(video_id)
                    st.session_state.transcript = transcript

                with st.spinner("⚙️ Chunking transcript..."):
                    chunks = chunk_text(transcript)
                    st.session_state.chunks = chunks

                with st.spinner("⚙️ Creating vector DB..."):
                    vector_db = create_vector_db(chunks)
                    st.session_state.vector_db = vector_db
                    st.session_state.video_loaded = True

                st.success(f"✅ Video loaded! ({len(chunks)} chunks created)")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure the video has subtitles enabled!")

if st.session_state.video_loaded:
    st.divider()

    with st.expander("👁️ Preview Transcript"):
        st.write(st.session_state.transcript[:1000] + "...")

    st.markdown("### 🎯 Choose What To Do")

    tab1, tab2, tab3 = st.tabs([
        "📝 Summarize",
        "💬 Ask Questions",
        "🧠 Generate Quiz"
    ])

    with tab1:
        st.markdown("### 📝 Video Summary")
        if st.button("🚀 Summarize Video", use_container_width=True):
            with st.spinner("🤖 Summarizing..."):
                result = summarize_video(st.session_state.chunks)
            st.success("✅ Summary Ready!")
            st.divider()
            st.markdown(result)
            st.download_button(
                label="⬇️ Download Summary",
                data=result,
                file_name="summary.txt",
                mime="text/plain"
            )

    with tab2:
        st.markdown("### 💬 Ask Questions")
        question = st.text_input(
            "Your question:",
            placeholder="e.g. What is the main topic?"
        )
        if st.button("💬 Get Answer", use_container_width=True):
            if not question.strip():
                st.warning("⚠️ Please type a question!")
            else:
                with st.spinner("🔍 Searching + generating answer..."):
                    result = answer_question(
                        st.session_state.vector_db,
                        question
                    )
                st.success("✅ Answer Ready!")
                st.divider()
                st.markdown(result)

    with tab3:
        st.markdown("### 🧠 Test Your Knowledge!")
        if st.button("🧠 Generate Quiz", use_container_width=True):
            with st.spinner("🤖 Generating quiz..."):
                result = generate_quiz(st.session_state.chunks)
            st.success("✅ Quiz Ready!")
            st.divider()
            st.markdown(result)

st.divider()
st.caption("Built with LangChain + Groq + ChromaDB + HuggingFace 🚀")


# 
# streamlit run app.py