# 🤖 Teams AI System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![LM Studio](https://img.shields.io/badge/LM%20Studio-Local%20AI-brightgreen.svg)](https://lmstudio.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/gghimire2041/ai-teams)
[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/gghimire2041/ai-teams/releases)

> **A production-ready multi-agent AI system where specialized AI agents collaborate to develop software projects autonomously using local LM Studio inference.**

🎯 **Teams AI System** represents the next evolution of software development where AI agents work together like a real development team - writing production-quality code, creating comprehensive tests, managing deployments, and generating professional documentation automatically using locally hosted AI models.

✨ **Now with full JSON serialization support, enhanced agent management, and production-ready stability!**

## 🎯 Key Features

### 🚀 **Multi-Agent Collaboration**
- **🧑‍💻 Coder Agent**: Generates production-ready code with proper documentation, error handling, and security best practices
- **🧪 Tester Agent**: Creates comprehensive test suites with edge cases, mocking, and coverage analysis
- **🔧 Integrator Agent**: Manages CI/CD pipelines, containerization, and deployment automation
- **📚 Documenter Agent**: Generates API docs, user guides, README files, and technical specifications
- **👥 Custom Agents**: Create specialized agents with unique names and capabilities

### 🧠 **Advanced AI Integration**
- **🔗 LM Studio Integration**: Full support for local models with fallback responses
- **🔄 Multi-Endpoint Support**: Automatically tries multiple API endpoints for reliability
- **⚡ Optimized Performance**: Intelligent caching and connection pooling
- **🎛️ Configurable Parameters**: Temperature, max tokens, top-p, and model selection
- **🔒 Privacy First**: All AI processing happens locally on your machine
- **📱 Offline Operation**: No internet required once models are downloaded

### 🎯 **Intelligent Task Orchestration**
- **📋 Smart Assignment**: Keyword-based automatic agent selection
- **🔗 Dependency Management**: Tasks wait for prerequisites to complete
- **⚖️ Priority Queuing**: High-priority tasks get processed first
- **🔄 Real-time Execution**: Live task processing with status updates
- **🛡️ Error Recovery**: Graceful handling of failures with detailed logging
- **📊 Progress Tracking**: Visual dashboard showing all agent activities

### 💾 **Advanced Memory System**
- **⚡ Short-term Memory**: Session context for ongoing conversations
- **🧠 Long-term Memory**: FAISS vector database for persistent knowledge
- **📚 Episodic Memory**: Complete history of agent actions and decisions
- **🔍 Semantic Search**: Find relevant past experiences automatically
- **💡 Learning Capability**: Agents improve based on past interactions

### 🌐 **Production-Ready Web Interface**
- **📱 Responsive Dashboard**: Works on desktop, tablet, and mobile
- **⚡ Real-time Updates**: Live agent status and task progress
- **🎨 Modern UI**: Clean, intuitive interface with beautiful styling
- **🔧 Agent Management**: Create, delete, and monitor custom agents
- **📊 System Metrics**: Performance monitoring and resource usage
- **🎛️ Configuration Panel**: Easy environment and model management

### ✨ **Latest Enhancements (v2.0.0)**
- **🔧 Fixed JSON Serialization**: Complete resolution of enum and datetime serialization issues
- **📦 Enhanced Database Management**: Improved SQLite operations with proper error handling
- **🚀 Optimized Performance**: Better memory management and faster response times
- **🔒 Robust Error Handling**: Comprehensive exception management throughout the system
- **📋 Agent Lifecycle Management**: Full CRUD operations for agent management
- **🎯 Task Assignment Control**: Both automatic and manual agent assignment options
- **📊 Improved Logging**: Detailed logging for debugging and monitoring
- **🔄 Auto-recovery**: System automatically handles and recovers from common issues

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │ Task Orchestrator│    │   LM Studio     │
│                 │◄──►│                 │◄──►│   Local API     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ Coder Agent  │ │ Tester Agent │ │Integrator Agt│
            └──────────────┘ └──────────────┘ └──────────────┘
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                    ┌─────────────────────────┐
                    │    Memory Manager       │
                    │  ┌─────────┬─────────┐  │
                    │  │ SQLite  │ FAISS   │  │
                    │  │Database │Vector DB│  │
                    │  └─────────┴─────────┘  │
                    └─────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Git** (configured with user.name and user.email)
- **LM Studio** with a compatible model loaded
- **4GB RAM** minimum (8GB+ recommended)

### LM Studio Setup

1. **Download and Install LM Studio**
   - Visit [https://lmstudio.ai/](https://lmstudio.ai/)
   - Download and install for your operating system

2. **Download a Compatible Model**
   - Recommended: Any 7B-20B parameter model (e.g., GPT-OSS-20B, Llama 2, Mistral)
   - Download through LM Studio's model browser

3. **Start LM Studio Local Server**
   - In LM Studio, go to "Local Server" tab
   - Load your chosen model
   - Click "Start Server" 
   - Verify server is running on `http://localhost:1234`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gghimire2041/ai-teams.git
   cd ai-teams
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Option 1: Install from requirements.txt
   pip install -r requirements.txt
   
   # Option 2: Install manually
   pip install Flask>=3.0.0 numpy>=1.24.0 faiss-cpu>=1.7.0 requests>=2.31.0
   ```

4. **Configure environment** (optional)
   ```bash
   # Set LM Studio URL if different from default
   export LM_STUDIO_URL="http://localhost:1234"
   export AI_TEMPERATURE="0.7"
   export AI_MAX_TOKENS="2048"
   
   # On Windows:
   set LM_STUDIO_URL=http://localhost:1234
   set AI_TEMPERATURE=0.7
   set AI_MAX_TOKENS=2048
   ```

### Running the System

1. **Ensure LM Studio is running** with a model loaded on `http://localhost:1234`

2. **Start the Teams AI System**:
   ```bash
   python teams_ai_system.py
   ```

3. **Expected Output**:
   ```
   🚀 Starting Teams AI System...
   🤖 Connecting to LM Studio...
   ✅ LM Studio connection verified
   ✅ Teams AI System initialized with default agents and sample project
   🌐 Web interface available at: http://localhost:8081
   ```

4. **Open your browser** and navigate to **http://localhost:8081**

## 📋 Usage Guide

### Creating Projects

1. Navigate to the web dashboard
2. Fill in the "Create New Project" form:
   - **Project Name**: Descriptive name (e.g., "E-commerce API")
   - **Project Description**: Detailed requirements
3. Click "Create Project"

### Creating Tasks

Tasks are automatically assigned to appropriate agents based on keywords:

| Keywords | Agent Type | Example Task |
|----------|------------|--------------|
| code, implement, develop | **Coder** | "Implement user authentication system" |
| test, verify, validate | **Tester** | "Create unit tests for API endpoints" |
| integrate, deploy, merge | **Integrator** | "Deploy application to production" |
| document, readme, docs | **Documenter** | "Generate API documentation" |

### Creating Custom Agents

1. Use the "Create Custom Agent" form
2. Choose agent type and give it a unique name
3. The new agent will appear in your team roster
4. Assign tasks specifically to your custom agents

### Monitoring Progress

- **Agent Status**: See what each agent is currently working on
- **Task Progress**: Track completion status and outputs
- **System Metrics**: Monitor queue size and performance
- **LM Studio Status**: Verify AI model connectivity

## 🔧 Configuration

### Environment Variables

```bash
# LM Studio Configuration
export LM_STUDIO_URL="http://localhost:1234"    # Default LM Studio URL
export LM_STUDIO_MODEL="local-model"            # Model identifier
export LM_STUDIO_TIMEOUT="120"                  # Request timeout (seconds)

# AI Parameters
export AI_TEMPERATURE="0.7"                     # Creativity vs consistency
export AI_MAX_TOKENS="2048"                     # Maximum response length
export AI_TOP_P="0.9"                          # Nucleus sampling parameter

# System Configuration
export TEAMS_AI_PORT="8080"                     # Web interface port
export TEAMS_AI_HOST="0.0.0.0"                 # Web interface host
```

### Directory Structure

After running, the system creates:

```
teams-ai-system/
├── teams_ai_system.py          # Main application
├── templates/
│   └── dashboard.html          # Web interface
├── data/
│   ├── teams.db               # SQLite database
│   ├── memory/                # Vector database storage
│   └── repositories/          # Generated project repositories
└── requirements.txt           # Python dependencies
```

## 🛠️ Troubleshooting

### Common Issues

#### LM Studio Connection Problems

**Error**: `Cannot POST /v1/chat/completions`
**Solution**: 
1. Verify LM Studio is running on `http://localhost:1234`
2. Ensure a model is loaded (not just downloaded)
3. Check that the local server is started in LM Studio

**Error**: `LM Studio API error 404`
**Solution**:
1. Test LM Studio manually: visit `http://localhost:1234` in browser
2. Verify correct port - should be 1234, not 41343 or other ports
3. Set environment variable: `export LM_STUDIO_URL="http://localhost:1234"`

#### Model Performance Issues

**Problem**: Slow AI responses
**Solution**:
- Enable GPU acceleration in LM Studio if available
- Reduce `AI_MAX_TOKENS` for shorter responses
- Use a smaller model if hardware is limited

#### Memory Issues

**Problem**: System running out of memory
**Solution**:
- Close other applications
- Use a smaller AI model
- Restart the system periodically for long sessions

### Testing LM Studio Connection

```bash
# Test if LM Studio is responding
curl http://localhost:1234/v1/models

# Test a simple completion
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## 📊 Example Workflows

### Web Application Development

```python
# Create project through web interface:
# Project: "Task Management App"
# Description: "Build a React/Node.js task management application"

# The system will automatically create and execute tasks like:
# 1. "Database Schema Design" → Coder Agent
# 2. "User Authentication API" → Coder Agent  
# 3. "Frontend Components" → Coder Agent
# 4. "API Testing Suite" → Tester Agent
# 5. "Integration Tests" → Tester Agent
# 6. "Production Deployment" → Integrator Agent
# 7. "User Documentation" → Documenter Agent
# 8. "API Documentation" → Documenter Agent
```

### API Development

```python
# Project: "REST API for Inventory System"
# Tasks are created with dependencies:
# 1. Database Models → Tests for Models → API Endpoints → API Tests → Documentation → Deployment
```

## 🔒 Security & Privacy

### Local AI Advantages

- **Complete Privacy**: All AI processing happens locally
- **No API Keys**: No external API credentials required
- **Offline Operation**: Works without internet connection
- **Data Control**: Your code and data never leave your machine

### Security Best Practices

- Keep LM Studio updated to latest version
- Run the system in a virtual environment
- Regularly backup your projects and data
- Monitor system resources during operation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different LM Studio models
5. Submit a pull request

## 📚 API Reference

### REST Endpoints

```
GET  /api/status              # System and LM Studio status
GET  /api/tasks               # List all tasks
GET  /api/agents              # List all agents
GET  /api/available_agents    # Available agents for assignment
GET  /api/memory/{agent_id}   # Agent memory and learning
POST /create_project          # Create new project
POST /create_task             # Create new task
POST /create_agent            # Create custom agent
POST /delete_agent            # Remove agent
```

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Documentation**: Read this README thoroughly
- **LM Studio Issues**: Check [LM Studio documentation](https://lmstudio.ai/)
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions in GitHub Discussions

## 🎯 Roadmap

- [ ] Support for more local AI platforms (Ollama, LocalAI)
- [ ] Advanced task dependency management
- [ ] Code review and approval workflows
- [ ] Plugin system for custom agent types
- [ ] Integration with popular IDEs
- [ ] Multi-project management
- [ ] Team collaboration features

---

**Ready to revolutionize your development workflow with local AI agents?**

🚀 **Get started now and watch your AI team build amazing projects!**