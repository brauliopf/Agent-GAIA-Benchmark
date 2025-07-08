# 🤖 Multimodal AI Agent Dashboard

A sophisticated multimodal AI agent built with LangGraph that can analyze images, videos, audio, documents, and perform web searches. Features an interactive Gradio dashboard for testing against GAIA benchmark questions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-Dashboard-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

### 🎯 Core Capabilities

- **Multimodal Processing**: Analyze images, videos, audio files, and documents
- **Web Intelligence**: Perform web searches and Wikipedia lookups
- **Code Execution**: Execute Python code dynamically with safety measures
- **Plan-and-Execute Architecture**: Intelligent task planning and execution workflow
- **GAIA Benchmark Testing**: Evaluate performance against standardized AI benchmarks

### 🛠️ Media Processing Tools

- **📸 Image Analysis**: Detailed image description and content analysis using GPT-4 Vision
- **🎥 Video Processing**: YouTube video analysis and content extraction
- **🎵 Audio Transcription**: Convert audio files to text using Whisper
- **📄 Document Reading**: Process PDF, Excel, and text files
- **🔍 Web Search**: Tavily search integration for real-time information

### 📊 Interactive Dashboard

- **General Q&A Interface**: Ask any question and get comprehensive answers
- **GAIA Benchmark Mode**: Select and test against specific benchmark questions
- **Real-time Results**: View detailed responses and processing logs
- **User Authentication**: Secure OAuth integration with Hugging Face

## 🏗️ Technology Stack

### **Core Framework**

- **[LangGraph](https://langchain-ai.github.io/langgraph/)**: State-based agent orchestration
- **[LangChain](https://langchain.com/)**: LLM application framework
- **[Gradio](https://gradio.app/)**: Interactive web interface

### **AI/ML Models**

- **OpenAI GPT-4**: Vision and text processing
- **Groq Whisper**: Audio transcription
- **Google Gemini**: Additional AI capabilities
- **Multiple LLM Providers**: Flexible model selection

### **Tools & Integrations**

- **Tavily Search**: Web search capabilities
- **Wikipedia API**: Knowledge base access
- **YouTube Processing**: Video content analysis
- **File Processing**: PDF, Excel, audio, image support

### **Infrastructure**

- **Python 3.8+**: Core runtime
- **Hugging Face Spaces**: Deployment platform
- **OAuth Authentication**: Secure user management
- **Temporary File Handling**: Safe file processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- API keys for OpenAI, Groq, and Tavily
- Hugging Face account (for authentication)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/hf_agents.git
   cd hf_agents
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   python src/app.py
   ```

### Environment Variables

```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
SPACE_ID=your_huggingface_space_id
```

## 🔧 Usage

### General Q&A Mode

1. Navigate to the "General Use" tab
2. Enter your question in the text area
3. Click "Submit Question" to get a comprehensive answer
4. View the detailed response and any processed media

### GAIA Benchmark Mode

1. Switch to the "GAIA Benchmark" tab
2. Click "Load Questions" to fetch available benchmark questions
3. Select questions you want to test
4. Click "Run Test on Selected Questions" to evaluate performance
5. Review results and submit answers for scoring

## 🏛️ Architecture

### Agent Workflow

```
Question Input → Planner → File Download (if needed) → Executor → Replanner → Final Answer
```

### Key Components

- **Planner**: Analyzes questions and creates execution plans
- **Executor**: Runs tasks using available tools and models
- **Replanner**: Validates outputs and decides on next steps
- **Tool Manager**: Handles file downloads and processing
- **State Management**: Maintains conversation context

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the excellent framework
- [Hugging Face](https://huggingface.co/) for hosting and authentication
- [OpenAI](https://openai.com/) for GPT-4 and vision capabilities
- [Gradio](https://gradio.app/) for the intuitive interface
- GAIA benchmark team for evaluation standards

---

Built with ❤️ using LangGraph and modern AI technologies
