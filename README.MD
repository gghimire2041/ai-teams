# Teams AI System 🤖

A fully working prototype of collaborative AI agents for software development, powered by LM Studio for local AI inference.

## Overview

Teams AI System is a multi-agent framework that simulates a software development team using AI agents. Each agent specializes in different aspects of software development:

- **🧑‍💻 Coder Agent**: Code generation, debugging, and refactoring
- **🧪 Tester Agent**: Unit testing, integration testing, and test automation
- **🔧 Integrator Agent**: Deployment, CI/CD, and system integration
- **📚 Documenter Agent**: Documentation, README generation, and user guides

## Features

### 🚀 Core Capabilities
- **Multi-Agent Collaboration**: Agents work together on complex software projects
- **Local AI Inference**: Powered by LM Studio for privacy and control
- **Task Orchestration**: Intelligent task assignment and dependency management
- **Version Control**: Built-in Git integration for project management
- **Memory Management**: Both short-term and long-term memory for agents
- **Web Interface**: User-friendly dashboard for project and agent management

### 🧠 AI Integration
- **LM Studio Compatibility**: Works with any model supported by LM Studio
- **Fallback Responses**: Graceful degradation when AI is unavailable
- **Configurable Parameters**: Adjustable temperature, max tokens, and other AI settings
- **Multiple Endpoint Support**: Automatic fallback across different LM Studio endpoints

### 📊 Project Management
- **Project Creation**: Initialize new projects with Git repositories
- **Task Dependencies**: Complex task dependency chains
- **Agent Assignment**: Manual or automatic agent assignment
- **Real-time Status**: Live updates on agent and task status
- **Memory Persistence**: Agent experiences saved across sessions

## Prerequisites

### Required Software
1. **Python 3.8+**
2. **LM Studio** - Download from [lmstudio.ai](https://lmstudio.ai)
3. **Git** - For version control functionality

### Python Dependencies
```bash
pip install flask requests numpy faiss-cpu
```

### LM Studio Setup
1. Download and install LM Studio
2. Download a compatible model (recommended: any 7B+ parameter model)
3. Load the model in LM Studio
4. Start the local server (typically runs on `http://localhost:1234`)

## Installation

### 1. Clone or Download
Save the `teams_ai_system.py` file to your desired directory.

### 2. Install Dependencies
```bash
pip install flask requests numpy faiss-cpu
```

### 3. Configure Environment (Optional)
Create a `.env` file or set environment variables:

```bash
# LM Studio Configuration
export LM_STUDIO_URL="http://localhost:1234"
export LM_STUDIO_MODEL="your-model-name"
export LM_STUDIO_TIMEOUT="120"

# AI Parameters
export AI_TEMPERATURE="0.7"
export AI_MAX_TOKENS="2048"
export AI_TOP_P="0.9"
```

### 4. Run the System
```bash
python teams_ai_system.py
```

The web interface will be available at: **http://localhost:8081**

## Quick Start Guide

### 1. Start LM Studio
- Open LM Studio
- Load your preferred model
- Start the local server

### 2. Launch Teams AI System
```bash
python teams_ai_system.py
```

### 3. Access Web Interface
Open your browser and navigate to `http://localhost:8081`

### 4. Create Your First Project
1. Use the "Create New Project" form
2. Enter a project name and description
3. Click "Create Project"

### 5. Add Custom Agents
1. Use the "Create Custom Agent" form
2. Choose an agent type and give it a name
3. Click "Create Agent"

### 6. Create Tasks
1. Use the "Create New Task" form
2. Specify task details and optionally assign to a specific agent
3. Watch as agents automatically work on the tasks

## Agent Types

### 🧑‍💻 Coder Agent
**Capabilities:**
- Code generation and implementation
- Debugging and troubleshooting
- Code refactoring and optimization
- Code review and best practices

**Use Cases:**
- "Implement user authentication system"
- "Create REST API endpoints"
- "Debug memory leak in application"

### 🧪 Tester Agent
**Capabilities:**
- Unit test creation
- Integration testing
- Test automation
- Coverage analysis

**Use Cases:**
- "Create tests for authentication module"
- "Verify API endpoint functionality"
- "Generate performance tests"

### 🔧 Integrator Agent
**Capabilities:**
- CI/CD pipeline setup
- Deployment configuration
- System integration
- Infrastructure management

**Use Cases:**
- "Set up Docker deployment"
- "Create GitHub Actions workflow"
- "Configure production environment"

### 📚 Documenter Agent
**Capabilities:**
- API documentation
- User guides and tutorials
- Code comments
- README generation

**Use Cases:**
- "Document REST API endpoints"
- "Create user installation guide"
- "Generate code documentation"

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │ Task Orchestrator│    │   LM Studio     │
│                 │◄──►│                 │◄──►│   (Local AI)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │   AI Agents     │
                    │  ┌─────────────┐ │
                    │  │   Coder     │ │
                    │  │   Tester    │ │
                    │  │ Integrator  │ │
                    │  │ Documenter  │ │
                    │  └─────────────┘ │
                    └─────────────────┘
                               │
                    ┌─────────────────┐
                    │ Data Persistence│
                    │  ┌─────────────┐ │
                    │  │  SQLite DB  │ │
                    │  │  Git Repos  │ │
                    │  │ Vector DB   │ │
                    │  └─────────────┘ │
                    └─────────────────┘
```

### Data Flow
1. **User Input**: Tasks and projects created via web interface
2. **Task Orchestration**: Central orchestrator manages task assignment
3. **Agent Execution**: Specialized agents process tasks using LM Studio
4. **Memory Storage**: Agent experiences and outputs stored persistently
5. **Version Control**: Code and documentation committed to Git repositories

## Configuration

### LM Studio Settings
The system automatically detects and connects to LM Studio using multiple endpoint attempts:
- `http://localhost:1234/v1/chat/completions`
- `http://localhost:1234/api/v1/chat/completions`
- `http://localhost:1234/chat/completions`
- And more...

### AI Parameters
- **Temperature**: Controls creativity (0.0-1.0, default: 0.7)
- **Max Tokens**: Maximum response length (default: 2048)
- **Top P**: Nucleus sampling parameter (default: 0.9)

### File Structure
```
teams_ai_system/
├── teams_ai_system.py          # Main application
├── data/
│   ├── teams.db               # SQLite database
│   ├── memory/                # Agent memory storage
│   └── repositories/          # Git repositories for projects
├── templates/
│   └── dashboard.html         # Web interface template
└── README.md                  # This file
```

## API Endpoints

### REST API
- `GET /api/status` - System status
- `GET /api/tasks` - List tasks
- `GET /api/agents` - List agents
- `GET /api/available_agents` - Available agents for assignment
- `GET /api/lm_studio_status` - LM Studio connection status
- `GET /api/memory/<agent_id>` - Agent memory

### Web Interface
- `POST /create_project` - Create new project
- `POST /create_agent` - Create custom agent
- `POST /create_task` - Create new task
- `POST /delete_agent` - Delete agent

## Troubleshooting

### LM Studio Connection Issues

**Problem**: "Cannot connect to LM Studio"
**Solutions:**
1. Ensure LM Studio is running
2. Verify a model is loaded
3. Check that the local server is started
4. Confirm the URL (default: `http://localhost:1234`)

**Problem**: "All LM Studio endpoints failed"
**Solutions:**
1. Restart LM Studio
2. Try a different model
3. Check firewall settings
4. Verify port availability

### General Issues

**Problem**: Tasks stuck in "pending" status
**Solutions:**
1. Check if agents are idle
2. Verify task dependencies are completed
3. Restart the system if needed

**Problem**: Database errors
**Solutions:**
1. Delete `data/teams.db` to reset
2. Ensure write permissions in data directory

**Problem**: Git operations failing
**Solutions:**
1. Ensure Git is installed and in PATH
2. Check repository permissions

## Development

### Adding New Agent Types
1. Inherit from `BaseAgent`
2. Implement required abstract methods
3. Register in `TaskOrchestrator.create_custom_agent()`

### Extending Functionality
- **New Task Types**: Modify task orchestration logic
- **Additional AI Models**: Extend `LMStudioClient` 
- **Custom Memory**: Enhance `MemoryManager`
- **Web Interface**: Modify templates and routes

## Performance Tips

### For Better AI Responses
1. Use specific, detailed task descriptions
2. Choose appropriate agent types for tasks
3. Set reasonable temperature values
4. Ensure model has sufficient context length

### For System Performance
1. Use SSD storage for repositories
2. Allocate sufficient RAM for LM Studio
3. Use models appropriate for your hardware
4. Monitor system resources

## Security Considerations

### Local AI Benefits
- **Privacy**: All AI processing happens locally
- **Data Control**: No external API calls for AI inference
- **Network Security**: Reduced external dependencies

### General Security
- Web interface runs on localhost by default
- No authentication system included (add if needed)
- Git repositories stored locally

## Contributing

### Areas for Improvement
- Authentication system for web interface
- More sophisticated agent communication
- Enhanced memory and context management
- Support for additional AI providers
- Advanced project templates
- Real-time collaboration features

### Development Setup
1. Fork or copy the code
2. Set up development environment
3. Make changes and test thoroughly
4. Consider submitting improvements

## License

This project is provided as-is for educational and development purposes. Modify and distribute according to your needs.

## Support

### Getting Help
1. Check the troubleshooting section
2. Verify LM Studio setup and model loading
3. Review system logs for error messages
4. Ensure all dependencies are installed

### Common Resources
- [LM Studio Documentation](https://lmstudio.ai)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Git Documentation](https://git-scm.com/doc)

---

**Built with ❤️ for the AI development community**

*Teams AI System - Where artificial intelligence meets collaborative software development*
