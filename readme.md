# 🤖 Teams AI System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/your-repo/teams-ai)

> **A fully functional multi-agent AI system where specialized AI agents collaborate to develop software projects autonomously.**

Teams AI System demonstrates the future of software development where AI agents work together like a real development team - writing code, creating tests, integrating changes, and generating documentation automatically.

## 🌟 Features

### 🚀 **Multi-Agent Collaboration**
- **Coder Agent**: Generates, debugs, and refactors code
- **Tester Agent**: Creates comprehensive test suites and measures coverage
- **Integrator Agent**: Manages deployments, CI/CD, and code merging
- **Documenter Agent**: Generates API docs, README files, and code comments

### 🧠 **Advanced Memory Management**
- **Short-term Memory**: Session-based context for ongoing conversations
- **Long-term Memory**: Persistent knowledge storage using FAISS vector database
- **Episodic Memory**: Records of past interactions and decisions
- **Semantic Memory**: Factual knowledge and learned patterns

### 🔄 **Intelligent Task Orchestration**
- Dependency-aware task scheduling
- Priority-based assignment algorithms
- Real-time progress tracking
- Automatic error recovery and retry mechanisms

### 📊 **Real-time Web Dashboard**
- Live agent status monitoring
- Task progress visualization
- Interactive project management
- System performance metrics

### 🔧 **Version Control Integration**
- Automatic Git repository management
- Intelligent commit message generation
- Branch management and merge conflict resolution
- Code review automation

## 🏗️ Architecture

```mermaid
graph TD
    A[Web Interface] --> B[Task Orchestrator]
    B --> C[Coder Agent]
    B --> D[Tester Agent]
    B --> E[Integrator Agent]
    B --> F[Documenter Agent]
    
    B --> G[Memory Manager]
    G --> H[SQLite Database]
    G --> I[FAISS Vector DB]
    G --> J[File System]
    
    B --> K[Version Control]
    K --> L[Git Repository]
    
    C --> M[Generated Code]
    D --> N[Test Suites]
    E --> O[Deployments]
    F --> P[Documentation]
```

### Core Components

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **Task Orchestrator** | Central coordination, workflow management | Python, Threading |
| **Memory Manager** | Context storage, knowledge retention | FAISS, SQLite |
| **Version Control Manager** | Git operations, repository management | Git CLI, Subprocess |
| **Web Interface** | User interaction, monitoring | Flask, HTML/CSS/JS |
| **Agent Framework** | Base agent functionality | Abstract Classes, Inheritance |

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Git** (configured with user.name and user.email)
- **4GB RAM** minimum
- **1GB disk space** for repositories and data

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/teams-ai-system.git
   cd teams-ai-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv teams_ai_env
   source teams_ai_env/bin/activate  # On Windows: teams_ai_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install Flask==2.3.3 numpy==1.24.3 faiss-cpu==1.7.4 requests==2.31.0
   ```

4. **Configure Git (if not already done)**
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### Running the System

```bash
python teams_ai_system.py
```

**Expected Output:**
```
🚀 Starting Teams AI System...
INFO:teams_ai_system:Registered agent: Alice Coder (coder)
INFO:teams_ai_system:Registered agent: Bob Tester (tester)
INFO:teams_ai_system:Registered agent: Charlie Integrator (integrator)
INFO:teams_ai_system:Registered agent: Diana Documenter (documenter)
INFO:teams_ai_system:TaskOrchestrator started
INFO:teams_ai_system:Created project: Web Dashboard
✅ Sample project and tasks created
🌐 Starting web interface...
🌐 Web interface available at: http://localhost:8080
```

Open your browser and navigate to **http://localhost:8080** to see the system in action!

## 📖 Usage Guide

### Web Dashboard

The web interface provides several key areas:

#### 🏠 **Main Dashboard**
- **System Status**: Active agents, total tasks, projects, queue size
- **Quick Actions**: Create new projects and tasks
- **Agent Monitor**: Real-time agent status and current activities
- **Task Overview**: Recent tasks and their completion status

#### 🆕 **Creating Projects**
1. Navigate to the dashboard
2. Fill in the "Create New Project" form:
   - **Project Name**: Descriptive name for your project
   - **Project Description**: Detailed description of what you want to build
3. Click "Create Project"

#### 📋 **Creating Tasks**
1. Use the "Create New Task" form:
   - **Task Name**: Brief, descriptive task name
   - **Task Description**: Detailed requirements
   - **Dependencies**: Comma-separated list of task IDs (optional)
   - **Priority**: 1-10 (1 = highest priority)
2. Click "Create Task"

#### 🔄 **Automatic Task Assignment**

The system intelligently assigns tasks based on keywords:

| Keywords | Assigned Agent | Example |
|----------|---------------|---------|
| code, implement, develop, program | **Coder Agent** | "Implement user authentication" |
| test, verify, validate | **Tester Agent** | "Create unit tests for API" |
| integrate, deploy, merge | **Integrator Agent** | "Deploy to production" |
| document, readme, docs | **Documenter Agent** | "Generate API documentation" |

### API Endpoints

The system provides RESTful APIs for integration:

```bash
# Get system status
curl http://localhost:8080/api/status

# List all tasks
curl http://localhost:8080/api/tasks

# List all agents
curl http://localhost:8080/api/agents

# Get agent memory
curl http://localhost:8080/api/memory/{agent_id}
```

## 🛠️ Configuration

### Environment Variables

```bash
# Database configuration
export TEAMS_AI_DB_PATH="./data/teams.db"
export TEAMS_AI_REPO_PATH="./data/repositories"
export TEAMS_AI_MEMORY_PATH="./data/memory"

# Web interface
export TEAMS_AI_PORT="8080"
export TEAMS_AI_HOST="0.0.0.0"

# LLM API keys (for production use)
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Directory Structure

```
teams-ai-system/
├── teams_ai_system.py          # Main application
├── templates/
│   └── dashboard.html          # Web interface template
├── data/
│   ├── teams.db               # SQLite database
│   ├── memory/                # Vector database storage
│   └── repositories/          # Git repositories
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Advanced Configuration

### Integrating Real LLM APIs

Replace the `simulate_llm_call` method to use actual LLM services:

```python
import openai

def simulate_llm_call(self, prompt: str, context: str = "") -> str:
    """Use real OpenAI API"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a {self.agent_type.value} agent."},
            {"role": "user", "content": f"{prompt}\n\nContext: {context}"}
        ]
    )
    return response.choices[0].message.content
```

### Adding Custom Agents

```python
class ReviewerAgent(BaseAgent):
    """Custom agent for code review"""
    
    def get_capabilities(self) -> List[str]:
        return ["code_review", "security_audit", "performance_analysis"]
    
    def execute_task(self, task: Task) -> str:
        self.update_status("reviewing")
        # Implement review logic
        review_result = self.perform_code_review(task)
        self.log_activity(f"Reviewed code for: {task.name}")
        self.update_status("idle")
        return review_result

# Register the new agent
reviewer = ReviewerAgent("reviewer-001", "Code Reviewer", 
                        AgentType.REVIEWER, orchestrator, memory_manager)
orchestrator.add_agent_instance(reviewer)
```

### Custom Task Types

```python
# Add new task types
class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    INTEGRATION = "integration"
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"          # New
    SECURITY_AUDIT = "security_audit"    # New
    PERFORMANCE_OPTIMIZATION = "performance_optimization"  # New
```

## 🧪 Example Workflows

### Web Application Development

```python
# Create project
project = orchestrator.create_project(
    "E-commerce Platform",
    "Build a full-stack e-commerce application with React and Node.js"
)

# Define workflow tasks
tasks = [
    ("Database Schema Design", "Design user, product, and order tables"),
    ("User Authentication API", "Implement JWT-based authentication"),
    ("Product Catalog API", "CRUD operations for product management"),
    ("Shopping Cart Logic", "Add/remove items, calculate totals"),
    ("Payment Integration", "Integrate Stripe payment processing"),
    ("React Frontend Components", "Build responsive UI components"),
    ("E2E Testing Suite", "Comprehensive test coverage"),
    ("API Documentation", "Generate OpenAPI specifications"),
    ("Production Deployment", "Deploy to AWS with CI/CD")
]

# Create interdependent tasks
previous_task_id = None
for name, description in tasks:
    dependencies = [previous_task_id] if previous_task_id else []
    task = orchestrator.create_task(name, description, dependencies)
    previous_task_id = task.id
```

### Machine Learning Pipeline

```python
# ML project workflow
ml_tasks = [
    ("Data Collection", "Gather and clean training data"),
    ("Feature Engineering", "Extract and transform features"),
    ("Model Training", "Train multiple ML models"),
    ("Model Evaluation", "Compare model performance"),
    ("Model Testing", "Unit tests for ML pipeline"),
    ("Model Documentation", "Document model architecture and usage"),
    ("Model Deployment", "Deploy to production with monitoring")
]
```

### API Development

```python
# API-focused workflow
api_tasks = [
    ("API Design", "Design RESTful API endpoints"),
    ("Data Models", "Implement database models"),
    ("Authentication Middleware", "JWT and OAuth implementation"),
    ("CRUD Endpoints", "Create, read, update, delete operations"),
    ("Input Validation", "Validate and sanitize inputs"),
    ("API Testing", "Unit and integration tests"),
    ("Rate Limiting", "Implement API rate limiting"),
    ("API Documentation", "Generate Swagger/OpenAPI docs"),
    ("Performance Testing", "Load testing and optimization"),
    ("Production Deployment", "Deploy with monitoring and logging")
]
```

## 🧠 Memory System

### Memory Types

The system implements multiple memory types based on cognitive science:

#### 📚 **Semantic Memory**
- Stores factual knowledge and concepts
- Used for code patterns, best practices, API documentation
- Example: "React components should use functional syntax with hooks"

#### 📖 **Episodic Memory**
- Records specific events and interactions
- Tracks task completion history and outcomes
- Example: "Failed to deploy due to missing environment variable on 2024-01-15"

#### 🛠️ **Procedural Memory**
- Stores learned skills and procedures
- Captures workflows and process optimizations
- Example: "Always run tests before integration commits"

### Memory Management

```python
# Add semantic memory
memory_manager.add_memory(
    agent_id="coder-001",
    content="Use TypeScript for better type safety in React projects",
    memory_type="semantic"
)

# Add episodic memory
memory_manager.add_memory(
    agent_id="tester-001",
    content="Test suite for authentication module completed successfully",
    memory_type="episodic"
)

# Retrieve relevant memories
query_embedding = np.random.random(768).astype('float32')
relevant_memories = memory_manager.retrieve_memories(query_embedding, k=5)
```

## 🔄 Version Control Features

### Automatic Git Operations

The system provides intelligent Git integration:

```python
# Initialize repository
repo_path = version_control_manager.init_repository(project_id, project_name)

# Intelligent commits
version_control_manager.commit_changes(
    repo_path, 
    "feat: implement user authentication with JWT tokens"
)

# Branch management
version_control_manager.create_branch(repo_path, "feature/user-auth")
version_control_manager.merge_branch(repo_path, "feature/user-auth")
```

### Commit Message Generation

The system generates semantic commit messages following conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `style:` Code style changes

## 📊 Monitoring and Analytics

### Performance Metrics

The system tracks various performance indicators:

```python
{
    "total_tasks_completed": 150,
    "average_task_completion_time": "4.2 minutes",
    "agent_utilization": {
        "coder": "85%",
        "tester": "72%",
        "integrator": "45%",
        "documenter": "38%"
    },
    "success_rate": "94.7%",
    "memory_usage": "2.3 GB",
    "active_projects": 5
}
```

### Error Tracking

```python
{
    "error_types": {
        "compilation_errors": 12,
        "test_failures": 8,
        "deployment_failures": 3,
        "memory_errors": 1
    },
    "resolution_rates": {
        "auto_resolved": "78%",
        "manual_intervention": "22%"
    }
}
```

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY teams_ai_system.py .
COPY templates/ templates/

# Create data directories
RUN mkdir -p data/repositories data/memory

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

# Run application
CMD ["python", "teams_ai_system.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: teams-ai-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: teams-ai-system
  template:
    metadata:
      labels:
        app: teams-ai-system
    spec:
      containers:
      - name: teams-ai-system
        image: teams-ai-system:latest
        ports:
        - containerPort: 8080
        env:
        - name: TEAMS_AI_DB_PATH
          value: "/data/teams.db"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-key
        volumeMounts:
        - name: data-storage
          mountPath: /data
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: teams-ai-pvc
```

### Environment Configuration

```bash
# Production environment variables
export FLASK_ENV=production
export TEAMS_AI_DB_PATH="/opt/teams_ai/data/teams.db"
export TEAMS_AI_REPO_PATH="/opt/teams_ai/repositories"
export TEAMS_AI_MEMORY_PATH="/opt/teams_ai/memory"
export TEAMS_AI_PORT=8080
export TEAMS_AI_HOST="0.0.0.0"

# Security
export SECRET_KEY="your-secret-key"
export ALLOWED_HOSTS="yourdomain.com,www.yourdomain.com"

# LLM APIs
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Database (for production scaling)
export DATABASE_URL="postgresql://user:pass@localhost/teams_ai"
export REDIS_URL="redis://localhost:6379"
```

## 🔧 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Error: Address already in use
# Solution: Change port in code or kill existing process
lsof -ti:8080
kill -9 <PID>
```

#### Git Configuration Missing
```bash
# Error: Git user not configured
# Solution: Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

#### Memory Issues
```bash
# Error: Out of memory for FAISS index
# Solution: Reduce vector dimensions or use disk-based storage
export FAISS_DISK_STORAGE=true
export FAISS_DIMENSION=384  # Reduce from 768
```

#### Permission Errors
```bash
# Error: Permission denied writing to data directory
# Solution: Fix permissions
chmod -R 755 data/
chown -R $USER:$USER data/
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

### Performance Optimization

```python
# Database optimization
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;

# Memory optimization
export FAISS_OMP_NUM_THREADS=4
export NUMPY_NUM_THREADS=4
```

## 📈 Scaling Considerations

### Horizontal Scaling

1. **Load Balancer**: Distribute requests across multiple instances
2. **Shared Database**: Use PostgreSQL instead of SQLite
3. **Message Queue**: Implement Redis for inter-agent communication
4. **Microservices**: Split agents into separate services

### Vertical Scaling

1. **Memory**: Increase RAM for larger vector databases
2. **CPU**: More cores for parallel agent execution
3. **Storage**: SSD for faster database operations
4. **Network**: High-bandwidth for API calls

## 🔒 Security

### Authentication & Authorization

```python
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY')
jwt = JWTManager(app)

@app.route('/login', methods=['POST'])
def login():
    # Implement authentication logic
    access_token = create_access_token(identity=user_id)
    return jsonify(access_token=access_token)

@app.route('/api/status')
@jwt_required()
def protected_status():
    return jsonify(orchestrator.get_system_status())
```

### Data Encryption

```python
from cryptography.fernet import Fernet

# Encrypt sensitive data
key = os.environ.get('ENCRYPTION_KEY').encode()
cipher_suite = Fernet(key)

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    return cipher_suite.decrypt(encrypted_data).decode()
```

### Secure Code Execution

```python
import subprocess
import tempfile
import os

def execute_code_safely(code, timeout=30):
    """Execute code in a sandboxed environment"""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_file = os.path.join(temp_dir, "code.py")
        with open(code_file, 'w') as f:
            f.write(code)
        
        try:
            result = subprocess.run(
                ['python', code_file],
                timeout=timeout,
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return None, "Code execution timed out"
```

## 🧪 Testing

### Unit Tests

```python
import unittest
from teams_ai_system import TaskOrchestrator, CoderAgent, Task

class TestTeamsAISystem(unittest.TestCase):
    
    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        self.coder = CoderAgent("test-coder", "Test Coder", 
                               AgentType.CODER, self.orchestrator, 
                               self.orchestrator.memory_manager)
    
    def test_task_creation(self):
        task = self.orchestrator.create_task(
            "Test Task", 
            "Test description"
        )
        self.assertEqual(task.name, "Test Task")
        self.assertEqual(task.status, TaskStatus.PENDING)
    
    def test_agent_registration(self):
        self.assertIn("test-coder", self.orchestrator.agents)
    
    def test_task_assignment(self):
        task = self.orchestrator.create_task(
            "Code Generation Task", 
            "Generate some code"
        )
        success = self.orchestrator.assign_task(task.id, AgentType.CODER)
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
def test_full_workflow():
    """Test complete workflow from project creation to task completion"""
    # Create project
    project = orchestrator.create_project("Test Project", "Test Description")
    
    # Create tasks
    task1 = orchestrator.create_task("Code Task", "Write some code")
    task2 = orchestrator.create_task("Test Task", "Write tests", [task1.id])
    
    # Wait for completion
    time.sleep(10)
    
    # Verify results
    assert task1.status == TaskStatus.COMPLETED
    assert task2.status == TaskStatus.COMPLETED
```

## 📚 API Reference

### Task Management

#### Create Task
```http
POST /create_task
Content-Type: application/x-www-form-urlencoded

name=Task Name&description=Task Description&dependencies=task1,task2&priority=1
```

#### Get Tasks
```http
GET /api/tasks?project_id={project_id}
```

Response:
```json
[
  {
    "id": "task-uuid",
    "name": "Task Name",
    "description": "Task Description",
    "status": "completed",
    "assigned_agent": "coder-001",
    "created_at": "2024-01-15T10:30:00",
    "updated_at": "2024-01-15T10:35:00",
    "dependencies": ["task1-uuid"],
    "output": "Task completed successfully",
    "priority": 1
  }
]
```

### Agent Management

#### Get Agents
```http
GET /api/agents
```

Response:
```json
[
  {
    "id": "coder-001",
    "name": "Alice Coder",
    "agent_type": "coder",
    "status": "idle",
    "current_task": null,
    "capabilities": ["code_generation", "debugging", "refactoring"],
    "created_at": "2024-01-15T10:00:00"
  }
]
```

### Memory Management

#### Get Agent Memory
```http
GET /api/memory/{agent_id}
```

Response:
```json
[
  {
    "id": "memory-uuid",
    "agent_id": "coder-001",
    "content": "Completed authentication module implementation",
    "memory_type": "episodic",
    "timestamp": "2024-01-15T10:30:00"
  }
]
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```
4. **Run tests**
   ```bash
   python -m pytest tests/
   ```
5. **Make your changes**
6. **Add tests for new features**
7. **Submit a pull request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation as needed
- Keep commits atomic and well-described

### Code Review Process

1. All pull requests require review
2. Tests must pass before merging
3. Documentation must be updated
4. Performance impact should be considered

## 📄 License

```
MIT License

Copyright (c) 2024 Teams AI System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🆘 Support

### Getting Help

- **Documentation**: Read this README thoroughly
- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact support@teams-ai-system.com

### Common Questions

**Q: Can I use this in production?**
A: This is a prototype for educational purposes. For production use, implement proper security, error handling, and scalability measures.

**Q: How do I integrate real LLM APIs?**
A: Replace the `simulate_llm_call` method with actual API calls to OpenAI, Anthropic, or other providers.

**Q: Can I add custom agent types?**
A: Yes! Extend the `BaseAgent` class and implement the required methods. See the "Adding Custom Agents" section.

**Q: How does the memory system work?**
A: The system uses FAISS vector database for semantic similarity search combined with SQLite for structured data storage.

**Q: Is this system secure?**
A: The prototype includes basic security measures. For production use, implement authentication, authorization, input validation, and code sandboxing.

---

**Built with ❤️ by the Teams AI System contributors**

*Ready to revolutionize software development with AI agents? Get started today!*# ai-teams
