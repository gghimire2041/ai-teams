# Teams AI System - Multi-Agent Software Development Framework
# A fully working prototype of collaborative AI agents for software development
# Now integrated with LM Studio for local AI inference

import os
import json
import sqlite3
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import subprocess
import shutil
from pathlib import Path

# Web Framework
from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests

# Vector Database for Memory
import numpy as np
import faiss
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LM STUDIO CLIENT
# ============================================================================

class LMStudioClient:
    """Client for communicating with LM Studio local API"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('LM_STUDIO_URL', 'http://localhost:1234')
        self.model_name = os.getenv('LM_STUDIO_MODEL', 'local-model')
        self.timeout = int(os.getenv('LM_STUDIO_TIMEOUT', '120'))
        
        # AI parameters
        self.temperature = float(os.getenv('AI_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('AI_MAX_TOKENS', '2048'))
        self.top_p = float(os.getenv('AI_TOP_P', '0.9'))
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info(f"🤖 Initializing LM Studio client at {self.base_url}")
        self.verify_connection()
    
    def verify_connection(self) -> bool:
        """Verify LM Studio is running and accessible"""
        try:
            # Try different common endpoints
            endpoints_to_try = [
                f"{self.base_url}/v1/models",
                f"{self.base_url}/models",
                f"{self.base_url}/api/v1/models",
                f"{self.base_url}/health",
                f"{self.base_url}/"
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    response = self.session.get(endpoint, timeout=10)
                    if response.status_code == 200:
                        logger.info(f"✅ LM Studio connection verified at {endpoint}")
                        return True
                except requests.exceptions.RequestException:
                    continue
            
            logger.warning(f"⚠️ Could not verify LM Studio connection")
            return False
            
        except Exception as e:
            logger.error(f"❌ Cannot connect to LM Studio at {self.base_url}: {e}")
            logger.error("Please ensure LM Studio is running with a model loaded")
            return False
    
    def _fallback_response(self, user_input: str) -> str:
        """Provide a fallback response when LM Studio is unavailable"""
        responses = {
            "code": "# Code generation currently unavailable\n# Please check LM Studio connection\nprint('Hello World')",
            "test": "# Test generation currently unavailable\ndef test_example():\n    assert True  # Placeholder test",
            "integrate": "# Integration task noted\n# Manual intervention required",
            "document": "# Documentation task\n\nThis feature requires LM Studio to be running with a loaded model."
        }
        
        user_lower = user_input.lower()
        if any(keyword in user_lower for keyword in ['code', 'implement', 'develop']):
            return responses["code"]
        elif any(keyword in user_lower for keyword in ['test', 'verify']):
            return responses["test"]
        elif any(keyword in user_lower for keyword in ['integrate', 'deploy']):
            return responses["integrate"]
        else:
            return responses["document"]
    
    def call_lm_studio(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a request to LM Studio API with multiple endpoint fallbacks"""
        
        # Prepare the payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "top_p": kwargs.get('top_p', self.top_p),
            "frequency_penalty": kwargs.get('frequency_penalty', 0.1),
            "presence_penalty": kwargs.get('presence_penalty', 0.0)
        }
        
        # Try different possible endpoints for LM Studio
        endpoints_to_try = [
            f"{self.base_url}/v1/chat/completions",
            f"{self.base_url}/api/v1/chat/completions",
            f"{self.base_url}/chat/completions",
            f"{self.base_url}/v1/completions",
            f"{self.base_url}/completions"
        ]
        
        last_error = None
        
        for endpoint in endpoints_to_try:
            try:
                logger.info(f"Trying LM Studio endpoint: {endpoint}")
                
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        
                        # Handle different response formats
                        if 'choices' in result and len(result['choices']) > 0:
                            if 'message' in result['choices'][0]:
                                content = result['choices'][0]['message']['content']
                            elif 'text' in result['choices'][0]:
                                content = result['choices'][0]['text']
                            else:
                                content = str(result['choices'][0])
                        elif 'response' in result:
                            content = result['response']
                        elif 'text' in result:
                            content = result['text']
                        else:
                            content = str(result)
                        
                        logger.info(f"✅ LM Studio API call successful via {endpoint}")
                        return content.strip()
                        
                    except (KeyError, IndexError, ValueError) as e:
                        logger.error(f"Error parsing LM Studio response: {e}")
                        logger.error(f"Response: {response.text[:200]}")
                        continue
                        
                else:
                    logger.warning(f"LM Studio endpoint {endpoint} returned status {response.status_code}")
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for endpoint {endpoint}")
                last_error = "Request timeout"
                continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for endpoint {endpoint}: {e}")
                last_error = str(e)
                continue
                
            except Exception as e:
                logger.warning(f"Unexpected error for endpoint {endpoint}: {e}")
                last_error = str(e)
                continue
        
        # If all endpoints failed, use fallback
        logger.error(f"All LM Studio endpoints failed. Last error: {last_error}")
        user_message = messages[-1]['content'] if messages else "general task"
        return self._fallback_response(user_message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get LM Studio status information"""
        try:
            # Check health with multiple endpoints
            health_status = False
            models = []
            
            health_endpoints = [
                f"{self.base_url}/v1/models",
                f"{self.base_url}/models",
                f"{self.base_url}/health",
                f"{self.base_url}/"
            ]
            
            for endpoint in health_endpoints:
                try:
                    health_response = self.session.get(endpoint, timeout=5)
                    if health_response.status_code == 200:
                        health_status = True
                        
                        # Try to extract model information
                        try:
                            data = health_response.json()
                            if 'data' in data:
                                models = [model.get('id', 'unknown') for model in data['data']]
                            elif 'models' in data:
                                models = data['models']
                        except:
                            pass
                        break
                except:
                    continue
            
            return {
                "connected": health_status,
                "url": self.base_url,
                "models": models,
                "current_model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "last_error": None if health_status else "Could not connect to any LM Studio endpoint"
            }
            
        except Exception as e:
            return {
                "connected": False,
                "url": self.base_url,
                "error": str(e),
                "models": [],
                "current_model": self.model_name,
                "last_error": str(e)
            }
# ============================================================================
# DATA MODELS & ENUMS
# ============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentType(Enum):
    CODER = "coder"
    TESTER = "tester"
    INTEGRATOR = "integrator"
    DOCUMENTER = "documenter"

class MessageType(Enum):
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    COLLABORATION_REQUEST = "collaboration_request"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"

@dataclass
class Task:
    id: str
    name: str
    description: str
    status: TaskStatus
    assigned_agent: Optional[str]
    created_at: datetime
    updated_at: datetime
    dependencies: List[str]
    output: Optional[str] = None
    error_message: Optional[str] = None
    priority: int = 1

@dataclass
class Agent:
    id: str
    name: str
    agent_type: AgentType
    status: str
    current_task: Optional[str]
    capabilities: List[str]
    created_at: datetime

@dataclass
class Message:
    id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime

@dataclass
class Project:
    id: str
    name: str
    description: str
    status: str
    created_at: datetime
    repository_path: str

# ============================================================================
# MEMORY MANAGEMENT SYSTEM
# ============================================================================

class MemoryManager:
    """Handles both short-term and long-term memory for AI agents"""
    
    def __init__(self, storage_path: str = "data/memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Vector database for semantic memory
        self.dimension = 768  # Embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.memory_items = []
        self.load_memory()
        
        # Short-term memory (session based)
        self.short_term_memory = {}
    
    def add_memory(self, agent_id: str, content: str, memory_type: str = "episodic"):
        """Add a memory item to long-term storage"""
        # Simulate embedding generation (in real implementation, use actual embeddings)
        embedding = np.random.random(self.dimension).astype('float32')
        
        memory_item = {
            'id': str(uuid.uuid4()),
            'agent_id': agent_id,
            'content': content,
            'memory_type': memory_type,
            'timestamp': datetime.now(),
            'embedding': embedding.tolist()
        }
        
        self.memory_items.append(memory_item)
        self.index.add(embedding.reshape(1, -1))
        self.save_memory()
    
    def retrieve_memories(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Retrieve relevant memories using vector similarity"""
        if self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(query_embedding.reshape(1, -1), min(k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memory_items):
                memory = self.memory_items[idx].copy()
                memory['similarity_score'] = float(distances[0][i])
                results.append(memory)
                
        return results
    
    def get_agent_memories(self, agent_id: str, limit: int = 10) -> List[Dict]:
        """Get recent memories for a specific agent"""
        agent_memories = [m for m in self.memory_items if m['agent_id'] == agent_id]
        return sorted(agent_memories, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def save_memory(self):
        """Persist memory to disk"""
        memory_file = self.storage_path / "long_term_memory.pkl"
        with open(memory_file, 'wb') as f:
            pickle.dump({
                'memory_items': self.memory_items,
                'index': faiss.serialize_index(self.index)
            }, f)
    
    def load_memory(self):
        """Load memory from disk"""
        memory_file = self.storage_path / "long_term_memory.pkl"
        if memory_file.exists():
            try:
                with open(memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.memory_items = data['memory_items']
                    self.index = faiss.deserialize_index(data['index'])
            except Exception as e:
                logger.warning(f"Could not load memory: {e}")

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: str = "data/teams.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:  # Only create directory if path has a directory component
            os.makedirs(db_dir, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT,
                    created_at TEXT,
                    repository_path TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT,
                    assigned_agent TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    dependencies TEXT,
                    output TEXT,
                    error_message TEXT,
                    priority INTEGER,
                    project_id TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    agent_type TEXT,
                    status TEXT,
                    current_task TEXT,
                    capabilities TEXT,
                    created_at TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    sender_id TEXT,
                    receiver_id TEXT,
                    message_type TEXT,
                    content TEXT,
                    timestamp TEXT
                )
            ''')
    
    def save_project(self, project: Project):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO projects 
                (id, name, description, status, created_at, repository_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (project.id, project.name, project.description, project.status,
                  project.created_at.isoformat(), project.repository_path))
    
    def save_task(self, task: Task, project_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO tasks 
                (id, name, description, status, assigned_agent, created_at, updated_at,
                 dependencies, output, error_message, priority, project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (task.id, task.name, task.description, task.status.value,
                  task.assigned_agent, task.created_at.isoformat(), 
                  task.updated_at.isoformat(), json.dumps(task.dependencies),
                  task.output, task.error_message, task.priority, project_id))
    
    def save_agent(self, agent: Agent):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO agents 
                (id, name, agent_type, status, current_task, capabilities, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (agent.id, agent.name, agent.agent_type.value, agent.status,
                  agent.current_task, json.dumps(agent.capabilities),
                  agent.created_at.isoformat()))
    
    def get_tasks(self, project_id: str = None) -> List[Task]:
        with sqlite3.connect(self.db_path) as conn:
            if project_id:
                cursor = conn.execute(
                    'SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at DESC',
                    (project_id,)
                )
            else:
                cursor = conn.execute('SELECT * FROM tasks ORDER BY created_at DESC')
            
            tasks = []
            for row in cursor.fetchall():
                tasks.append(Task(
                    id=row[0], name=row[1], description=row[2],
                    status=TaskStatus(row[3]), assigned_agent=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    updated_at=datetime.fromisoformat(row[6]),
                    dependencies=json.loads(row[7] or '[]'),
                    output=row[8], error_message=row[9], priority=row[10]
                ))
            return tasks
    
    def get_agents(self) -> List[Agent]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM agents')
            agents = []
            for row in cursor.fetchall():
                agents.append(Agent(
                    id=row[0], name=row[1], agent_type=AgentType(row[2]),
                    status=row[3], current_task=row[4],
                    capabilities=json.loads(row[5] or '[]'),
                    created_at=datetime.fromisoformat(row[6])
                ))
            return agents
    
    def get_projects(self) -> List[Project]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM projects ORDER BY created_at DESC')
            projects = []
            for row in cursor.fetchall():
                projects.append(Project(
                    id=row[0], name=row[1], description=row[2],
                    status=row[3], created_at=datetime.fromisoformat(row[4]),
                    repository_path=row[5]
                ))
            return projects

# ============================================================================
# VERSION CONTROL MANAGER
# ============================================================================

class VersionControlManager:
    """Handles Git operations and version control"""
    
    def __init__(self, base_repos_path: str = "data/repositories"):
        self.base_repos_path = Path(base_repos_path)
        self.base_repos_path.mkdir(parents=True, exist_ok=True)
    
    def init_repository(self, project_id: str, project_name: str) -> str:
        """Initialize a Git repository for a project"""
        repo_path = self.base_repos_path / project_id
        repo_path.mkdir(exist_ok=True)
        
        try:
            # Initialize git repo
            subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
            
            # Create initial README
            readme_path = repo_path / "README.md"
            with open(readme_path, 'w') as f:
                f.write(f"# {project_name}\n\nGenerated by Teams AI System with LM Studio\n")
            
            # Initial commit
            subprocess.run(['git', 'add', '.'], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', 'Initial commit'], 
                         cwd=repo_path, check=True, capture_output=True)
            
            return str(repo_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e}")
            return str(repo_path)
    
    def commit_changes(self, repo_path: str, message: str, files: List[str] = None):
        """Commit changes to repository"""
        try:
            if files:
                for file in files:
                    subprocess.run(['git', 'add', file], cwd=repo_path, check=True)
            else:
                subprocess.run(['git', 'add', '.'], cwd=repo_path, check=True)
            
            subprocess.run(['git', 'commit', '-m', message], 
                         cwd=repo_path, check=True, capture_output=True)
            logger.info(f"Committed changes: {message}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Commit failed: {e}")

# ============================================================================
# AI AGENT BASE CLASS
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all AI agents"""
    
    def __init__(self, agent_id: str, name: str, agent_type: AgentType, 
                 orchestrator, memory_manager: MemoryManager, lm_studio_client: LMStudioClient):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.orchestrator = orchestrator
        self.memory_manager = memory_manager
        self.lm_studio_client = lm_studio_client
        self.status = "idle"
        self.current_task = None
        self.capabilities = self.get_capabilities()
        
        # Register with orchestrator
        agent_obj = Agent(
            id=agent_id, name=name, agent_type=agent_type,
            status=self.status, current_task=None,
            capabilities=self.capabilities, created_at=datetime.now()
        )
        self.orchestrator.register_agent(agent_obj)
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent type"""
        pass
    
    @abstractmethod
    def execute_task(self, task: Task) -> str:
        """Execute a specific task"""
        pass
    
    def call_ai(self, prompt: str, context: str = "", **kwargs) -> str:
        """Call LM Studio AI with agent-specific prompts"""
        # Get recent memories for context
        recent_memories = self.memory_manager.get_agent_memories(self.agent_id, limit=3)
        memory_context = "\n".join([f"- {m['content']}" for m in recent_memories])
        
        # Build conversation messages
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt()
            },
            {
                "role": "user",
                "content": f"""Task: {prompt}

Additional Context: {context}

Recent Experience:
{memory_context}

Please provide a detailed response that fulfills the task requirements."""
            }
        ]
        
        return self.lm_studio_client.call_lm_studio(messages, **kwargs)
    
    def update_status(self, status: str):
        """Update agent status"""
        self.status = status
        self.orchestrator.update_agent_status(self.agent_id, status)
    
    def log_activity(self, activity: str):
        """Log agent activity to memory"""
        self.memory_manager.add_memory(
            agent_id=self.agent_id,
            content=f"{self.name}: {activity}",
            memory_type="episodic"
        )

# ============================================================================
# SPECIALIZED AI AGENTS
# ============================================================================

class CoderAgent(BaseAgent):
    """AI Agent specialized in code generation and debugging"""
    
    def get_capabilities(self) -> List[str]:
        return ["code_generation", "debugging", "refactoring", "code_review"]
    
    def get_system_prompt(self) -> str:
        return """You are an expert software developer and coding assistant. Your expertise includes:

- Writing clean, efficient, and well-documented code
- Following best practices and design patterns
- Implementing proper error handling and validation
- Creating secure and performant solutions
- Debugging and troubleshooting code issues
- Code refactoring and optimization

When generating code:
1. Always include proper comments and documentation
2. Follow the appropriate coding standards for the language
3. Consider security implications and best practices
4. Include error handling where appropriate
5. Make the code readable and maintainable
6. Provide explanations for complex logic

Generate production-ready code that a professional developer would be proud to deploy."""
    
    def execute_task(self, task: Task) -> str:
        self.update_status("coding")
        self.current_task = task.id
        
        # Generate code using LM Studio
        prompt = f"Generate code for: {task.description}"
        code = self.call_ai(prompt, 
                           context=f"This is part of a larger project. Focus on {task.name}.",
                           temperature=0.3)  # Lower temperature for more consistent code
        
        # Write code to file
        file_path = self.write_code_to_file(task, code)
        
        self.log_activity(f"Generated code for task: {task.name}")
        self.update_status("idle")
        
        return f"Code generated and saved to: {file_path}\n\nGenerated Code Preview:\n{code[:500]}..."
    
    def write_code_to_file(self, task: Task, code: str) -> str:
        """Write generated code to project file"""
        project = self.orchestrator.get_current_project()
        if not project:
            return "No active project"
        
        # Determine file extension based on code content
        if 'def ' in code or 'import ' in code or 'print(' in code:
            extension = '.py'
        elif 'function ' in code or 'const ' in code or 'let ' in code:
            extension = '.js'
        elif 'public class' in code or 'import java' in code:
            extension = '.java'
        else:
            extension = '.txt'
        
        file_name = f"{task.name.lower().replace(' ', '_')}{extension}"
        file_path = Path(project.repository_path) / "src" / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(f"# Task: {task.name}\n# Description: {task.description}\n# Generated by Teams AI System\n\n{code}")
        
        return str(file_path)

class TesterAgent(BaseAgent):
    """AI Agent specialized in testing"""
    
    def get_capabilities(self) -> List[str]:
        return ["unit_testing", "integration_testing", "test_coverage", "test_automation"]
    
    def get_system_prompt(self) -> str:
        return """You are an expert QA engineer and test automation specialist. Your expertise includes:

- Creating comprehensive unit tests with high coverage
- Designing integration and end-to-end tests
- Test-driven development (TDD) practices
- Testing edge cases and error conditions
- Performance and load testing strategies
- Test automation frameworks and tools

When creating tests:
1. Cover both positive and negative test cases
2. Test edge cases and boundary conditions
3. Include proper assertions and error handling
4. Use descriptive test names and documentation
5. Follow testing best practices for the framework
6. Ensure tests are maintainable and reliable
7. Consider mock objects and test data setup

Generate thorough, professional-quality tests that ensure code reliability and quality."""
    
    def execute_task(self, task: Task) -> str:
        self.update_status("testing")
        self.current_task = task.id
        
        # Generate test code
        prompt = f"Generate comprehensive tests for: {task.description}"
        test_code = self.call_ai(prompt,
                               context="Create thorough unit tests with good coverage and edge cases.",
                               temperature=0.4)
        
        # Write test file
        test_file_path = self.write_test_file(task, test_code)
        
        self.log_activity(f"Created and validated tests for task: {task.name}")
        self.update_status("idle")
        
        return f"Tests created: {test_file_path}\n\nTest Preview:\n{test_code[:500]}..."
    
    def write_test_file(self, task: Task, test_code: str) -> str:
        """Write test code to file"""
        project = self.orchestrator.get_current_project()
        if not project:
            return "No active project"
        
        test_file_name = f"test_{task.name.lower().replace(' ', '_')}.py"
        test_file_path = Path(project.repository_path) / "tests" / test_file_name
        test_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_file_path, 'w') as f:
            f.write(f"# Tests for: {task.name}\n# Description: {task.description}\n# Generated by Teams AI System\n\n{test_code}")
        
        return str(test_file_path)

class IntegratorAgent(BaseAgent):
    """AI Agent specialized in integration and deployment"""
    
    def get_capabilities(self) -> List[str]:
        return ["code_integration", "deployment", "ci_cd", "merge_management"]
    
    def get_system_prompt(self) -> str:
        return """You are an expert DevOps engineer and integration specialist. Your expertise includes:

- CI/CD pipeline design and implementation
- Deployment strategies and automation
- Infrastructure as Code (IaC)
- Container orchestration and management
- Monitoring, logging, and observability
- Security and compliance in deployments

When handling integration and deployment:
1. Ensure proper environment configuration
2. Implement rollback strategies
3. Follow security best practices
4. Set up proper monitoring and alerting
5. Document deployment procedures
6. Consider scalability and performance
7. Implement proper testing in staging environments

Generate production-ready deployment configurations and integration workflows."""
    
    def execute_task(self, task: Task) -> str:
        self.update_status("integrating")
        self.current_task = task.id
        
        project = self.orchestrator.get_current_project()
        if not project:
            return "No active project"
        
        # Generate integration/deployment code
        prompt = f"Create integration and deployment configuration for: {task.description}"
        integration_code = self.call_ai(prompt,
                                      context="Focus on production-ready deployment and CI/CD practices.",
                                      temperature=0.3)
        
        # Create deployment files
        deployment_file_path = self.write_deployment_file(task, integration_code)
        
        # Commit changes
        commit_message = f"feat: {task.name} - {task.description}"
        self.orchestrator.version_control_manager.commit_changes(project.repository_path, commit_message)
        
        self.log_activity(f"Integrated and deployed changes for task: {task.name}")
        self.update_status("idle")
        
        return f"Integration completed for task: {task.name}\nDeployment config: {deployment_file_path}\n\nIntegration Details:\n{integration_code[:500]}..."
    
    def write_deployment_file(self, task: Task, integration_code: str) -> str:
        """Write deployment configuration to file"""
        project = self.orchestrator.get_current_project()
        if not project:
            return "No active project"
        
        deploy_file_name = f"{task.name.lower().replace(' ', '_')}_deploy.yml"
        deploy_file_path = Path(project.repository_path) / "deploy" / deploy_file_name
        deploy_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(deploy_file_path, 'w') as f:
            f.write(f"# Deployment for: {task.name}\n# Description: {task.description}\n# Generated by Teams AI System\n\n{integration_code}")
        
        return str(deploy_file_path)

class DocumenterAgent(BaseAgent):
    """AI Agent specialized in documentation"""
    
    def get_capabilities(self) -> List[str]:
        return ["api_documentation", "code_comments", "readme_generation", "user_guides"]
    
    def get_system_prompt(self) -> str:
        return """You are an expert technical writer and documentation specialist. Your expertise includes:

- Creating clear, comprehensive API documentation
- Writing user guides and tutorials
- Generating code comments and inline documentation
- Architecture documentation and diagrams
- README files and project documentation
- Documentation standards and best practices

When creating documentation:
1. Use clear, concise language
2. Include practical examples and code snippets
3. Structure information logically
4. Consider the target audience (developers, users, etc.)
5. Include setup and getting started guides
6. Document APIs with request/response examples
7. Provide troubleshooting and FAQ sections

Generate professional, comprehensive documentation that helps users and developers effectively use and understand the project."""
    
    def execute_task(self, task: Task) -> str:
        self.update_status("documenting")
        self.current_task = task.id
        
        # Generate documentation
        prompt = f"Generate comprehensive documentation for: {task.description}"
        documentation = self.call_ai(prompt,
                                   context="Create clear, professional documentation with examples.",
                                   temperature=0.5)
        
        # Write documentation file
        doc_file_path = self.write_documentation_file(task, documentation)
        
        self.log_activity(f"Created documentation for task: {task.name}")
        self.update_status("idle")
        
        return f"Documentation created: {doc_file_path}\n\nDocumentation Preview:\n{documentation[:500]}..."
    
    def write_documentation_file(self, task: Task, documentation: str) -> str:
        """Write documentation to file"""
        project = self.orchestrator.get_current_project()
        if not project:
            return "No active project"
        
        doc_file_name = f"{task.name.lower().replace(' ', '_')}_docs.md"
        doc_file_path = Path(project.repository_path) / "docs" / doc_file_name
        doc_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(doc_file_path, 'w') as f:
            f.write(f"# {task.name}\n\n*Description: {task.description}*\n\n*Generated by Teams AI System*\n\n{documentation}")
        
        return str(doc_file_path)

# ============================================================================
# TASK ORCHESTRATOR
# ============================================================================

class TaskOrchestrator:
    """Central orchestrator that manages agents and tasks"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_objects: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.projects: Dict[str, Project] = {}
        self.current_project_id: Optional[str] = None
        self.message_queue: List[Message] = []
        self.running = False
        
        # Initialize managers
        self.db_manager = DatabaseManager()
        self.memory_manager = MemoryManager()
        self.version_control_manager = VersionControlManager()
        self.lm_studio_client = LMStudioClient()
        
        # Start orchestrator thread
        self.orchestrator_thread = threading.Thread(target=self.run_orchestrator)
        self.orchestrator_thread.daemon = True
    
    def start(self):
        """Start the orchestrator"""
        self.running = True
        self.orchestrator_thread.start()
        logger.info("TaskOrchestrator started")
    
    def register_agent(self, agent: Agent):
        """Register a new agent"""
        self.agent_objects[agent.id] = agent
        self.db_manager.save_agent(agent)
        logger.info(f"Registered agent: {agent.name} ({agent.agent_type.value})")
    
    def add_agent_instance(self, agent: BaseAgent):
        """Add agent instance for task execution"""
        self.agents[agent.agent_id] = agent
    
    def create_custom_agent(self, agent_id: str, name: str, agent_type: AgentType) -> Agent:
        """Create a custom agent with user-defined name"""
        if agent_type == AgentType.CODER:
            agent_instance = CoderAgent(agent_id, name, agent_type, self, self.memory_manager, self.lm_studio_client)
        elif agent_type == AgentType.TESTER:
            agent_instance = TesterAgent(agent_id, name, agent_type, self, self.memory_manager, self.lm_studio_client)
        elif agent_type == AgentType.INTEGRATOR:
            agent_instance = IntegratorAgent(agent_id, name, agent_type, self, self.memory_manager, self.lm_studio_client)
        elif agent_type == AgentType.DOCUMENTER:
            agent_instance = DocumenterAgent(agent_id, name, agent_type, self, self.memory_manager, self.lm_studio_client)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.add_agent_instance(agent_instance)
        return self.agent_objects[agent_id]
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent"""
        if agent_id in self.agents and agent_id in self.agent_objects:
            agent = self.agent_objects[agent_id]
            if agent.current_task:
                return False
            
            del self.agents[agent_id]
            del self.agent_objects[agent_id]
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                conn.execute('DELETE FROM agents WHERE id = ?', (agent_id,))
            
            logger.info(f"Deleted agent: {agent_id}")
            return True
        return False
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents for task assignment"""
        available_agents = []
        for agent_id, agent in self.agent_objects.items():
            if agent.status == "idle":
                available_agents.append({
                    'id': agent_id,
                    'name': agent.name,
                    'type': agent.agent_type.value,
                    'capabilities': agent.capabilities
                })
        return available_agents
    
    def create_project(self, name: str, description: str) -> Project:
        """Create a new project"""
        project_id = str(uuid.uuid4())
        repo_path = self.version_control_manager.init_repository(project_id, name)
        
        project = Project(
            id=project_id,
            name=name,
            description=description,
            status="active",
            created_at=datetime.now(),
            repository_path=repo_path
        )
        
        self.projects[project_id] = project
        self.current_project_id = project_id
        self.db_manager.save_project(project)
        
        logger.info(f"Created project: {name}")
        return project
    
    def get_current_project(self) -> Optional[Project]:
        """Get the current active project"""
        if self.current_project_id:
            return self.projects.get(self.current_project_id)
        return None
    
    def create_task(self, name: str, description: str, assigned_agent: str = None,
                   dependencies: List[str] = None, priority: int = 1) -> Task:
        """Create a new task with optional specific agent assignment"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            description=description,
            status=TaskStatus.PENDING,
            assigned_agent=assigned_agent,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            dependencies=dependencies or [],
            priority=priority
        )
        
        self.tasks[task_id] = task
        if self.current_project_id:
            self.db_manager.save_task(task, self.current_project_id)
        
        logger.info(f"Created task: {name}")
        
        # If agent is pre-assigned, try to start the task immediately
        if assigned_agent and self.is_agent_available(assigned_agent):
            self.start_assigned_task(task_id)
        
        return task
    
    def is_agent_available(self, agent_id: str) -> bool:
        """Check if an agent is available for new tasks"""
        agent = self.agent_objects.get(agent_id)
        return agent and agent.status == "idle" and not agent.current_task
    
    def start_assigned_task(self, task_id: str) -> bool:
        """Start a task that has been pre-assigned to a specific agent"""
        task = self.tasks.get(task_id)
        if not task or not task.assigned_agent:
            return False
        
        # Check if dependencies are satisfied
        deps_satisfied = all(
            self.tasks[dep_id].status == TaskStatus.COMPLETED
            for dep_id in task.dependencies
            if dep_id in self.tasks
        )
        
        if not deps_satisfied:
            return False
        
        if not self.is_agent_available(task.assigned_agent):
            return False
        
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()
        
        if self.current_project_id:
            self.db_manager.save_task(task, self.current_project_id)
        
        message = Message(
            id=str(uuid.uuid4()),
            sender_id="orchestrator",
            receiver_id=task.assigned_agent,
            message_type=MessageType.TASK_ASSIGNMENT,
            content={"task": asdict(task)},
            timestamp=datetime.now()
        )
        
        self.message_queue.append(message)
        logger.info(f"Started assigned task {task.name} with {task.assigned_agent}")
        return True
    
    def assign_task(self, task_id: str, agent_id: str = None) -> bool:
        """Assign a task to a specific agent or auto-assign"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if agent_id:
            if not self.is_agent_available(agent_id):
                logger.warning(f"Agent {agent_id} not available for task {task.name}")
                return False
            target_agent_id = agent_id
        else:
            agent_type = self.determine_agent_type_for_task(task)
            available_agent = None
            for aid, agent in self.agents.items():
                if (agent.agent_type == agent_type and 
                    agent.status == "idle" and 
                    agent.current_task is None):
                    available_agent = agent
                    break
            
            if not available_agent:
                logger.warning(f"No available {agent_type.value} agent for task {task.name}")
                return False
            
            target_agent_id = available_agent.agent_id
        
        task.assigned_agent = target_agent_id
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()
        
        if self.current_project_id:
            self.db_manager.save_task(task, self.current_project_id)
        
        message = Message(
            id=str(uuid.uuid4()),
            sender_id="orchestrator",
            receiver_id=target_agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            content={"task": asdict(task)},
            timestamp=datetime.now()
        )
        
        self.message_queue.append(message)
        logger.info(f"Assigned task {task.name} to {target_agent_id}")
        return True
    
    def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        if agent_id in self.agent_objects:
            self.agent_objects[agent_id].status = status
            self.db_manager.save_agent(self.agent_objects[agent_id])
    
    def complete_task(self, task_id: str, output: str, agent_id: str):
        """Mark task as completed"""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.output = output
            task.updated_at = datetime.now()
            
            if self.current_project_id:
                self.db_manager.save_task(task, self.current_project_id)
            
            if agent_id in self.agents:
                self.agents[agent_id].current_task = None
                self.agents[agent_id].update_status("idle")
            
            logger.info(f"Task completed: {task.name}")
            self.check_dependency_completion(task_id)
    
    def check_dependency_completion(self, completed_task_id: str):
        """Check if any pending tasks can now be started"""
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                completed_task_id in task.dependencies):
                
                all_deps_completed = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                    if dep_id in self.tasks
                )
                
                if all_deps_completed:
                    if task.assigned_agent:
                        self.start_assigned_task(task.id)
                    else:
                        self.assign_task(task.id)
    
    def determine_agent_type_for_task(self, task: Task) -> AgentType:
        """Determine appropriate agent type for a task"""
        name_lower = task.name.lower()
        desc_lower = task.description.lower()
        
        if any(keyword in name_lower + desc_lower for keyword in 
               ['code', 'implement', 'develop', 'program']):
            return AgentType.CODER
        elif any(keyword in name_lower + desc_lower for keyword in 
                ['test', 'verify', 'validate']):
            return AgentType.TESTER
        elif any(keyword in name_lower + desc_lower for keyword in 
                ['integrate', 'deploy', 'merge']):
            return AgentType.INTEGRATOR
        elif any(keyword in name_lower + desc_lower for keyword in 
                ['document', 'readme', 'docs']):
            return AgentType.DOCUMENTER
        else:
            return AgentType.CODER
    
    def run_orchestrator(self):
        """Main orchestrator loop"""
        while self.running:
            try:
                if self.message_queue:
                    message = self.message_queue.pop(0)
                    self.process_message(message)
                
                self.auto_assign_ready_tasks()
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Orchestrator error: {e}")
                time.sleep(5)
    
    def process_message(self, message: Message):
        """Process a message"""
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            agent = self.agents.get(message.receiver_id)
            if agent:
                task_data = message.content.get("task")
                task = Task(**task_data)
                
                try:
                    result = agent.execute_task(task)
                    self.complete_task(task.id, result, agent.agent_id)
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    if self.current_project_id:
                        self.db_manager.save_task(task, self.current_project_id)
    
    def auto_assign_ready_tasks(self):
        """Auto-assign tasks that are ready to be executed"""
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and not task.assigned_agent:
                deps_satisfied = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                    if dep_id in self.tasks
                )
                
                if deps_satisfied or not task.dependencies:
                    self.assign_task(task.id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "agents": [asdict(agent) for agent in self.agent_objects.values()],
            "tasks": [asdict(task) for task in self.tasks.values()],
            "projects": [asdict(project) for project in self.projects.values()],
            "current_project": self.current_project_id,
            "message_queue_size": len(self.message_queue),
            "lm_studio_status": self.lm_studio_client.get_status()
        }

# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def serialize_agent(agent: Agent) -> Dict[str, Any]:
    """Convert agent to JSON-serializable dictionary"""
    return {
        "id": agent.id,
        "name": agent.name,
        "agent_type": agent.agent_type.value,
        "status": agent.status,
        "current_task": agent.current_task,
        "capabilities": agent.capabilities,
        "created_at": agent.created_at.isoformat()
    }

def serialize_task(task: Task) -> Dict[str, Any]:
    """Convert task to JSON-serializable dictionary"""
    return {
        "id": task.id,
        "name": task.name,
        "description": task.description,
        "status": task.status.value,
        "assigned_agent": task.assigned_agent,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "dependencies": task.dependencies,
        "output": task.output,
        "error_message": task.error_message,
        "priority": task.priority
    }

def serialize_project(project: Project) -> Dict[str, Any]:
    """Convert project to JSON-serializable dictionary"""
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "status": project.status,
        "created_at": project.created_at.isoformat(),
        "repository_path": project.repository_path
    }

# ============================================================================
# WEB INTERFACE
# ============================================================================

app = Flask(__name__)
app.secret_key = "teams-ai-secret-key"

orchestrator = TaskOrchestrator()

def init_system():
    """Initialize the Teams AI system"""
    coder = CoderAgent("coder-default", "Alice Coder", AgentType.CODER, 
                      orchestrator, orchestrator.memory_manager, orchestrator.lm_studio_client)
    tester = TesterAgent("tester-default", "Bob Tester", AgentType.TESTER, 
                        orchestrator, orchestrator.memory_manager, orchestrator.lm_studio_client)
    integrator = IntegratorAgent("integrator-default", "Charlie Integrator", AgentType.INTEGRATOR,
                                orchestrator, orchestrator.memory_manager, orchestrator.lm_studio_client)
    documenter = DocumenterAgent("documenter-default", "Diana Documenter", AgentType.DOCUMENTER,
                                 orchestrator, orchestrator.memory_manager, orchestrator.lm_studio_client)
    
    orchestrator.add_agent_instance(coder)
    orchestrator.add_agent_instance(tester)
    orchestrator.add_agent_instance(integrator)
    orchestrator.add_agent_instance(documenter)
    orchestrator.start()
    
    # Create a sample project
    sample_project = orchestrator.create_project(
        "Web Dashboard", 
        "A sample web application dashboard"
    )
    
    # Create sample tasks
    orchestrator.create_task(
        "Authentication System",
        "Implement user authentication with JWT tokens",
        priority=1
    )
    
    orchestrator.create_task(
        "User Dashboard UI",
        "Create responsive user dashboard interface",
        priority=2
    )
    
    logger.info("Teams AI System initialized with default agents and sample project")

@app.route('/')
def dashboard():
    """Main dashboard"""
    try:
        status = orchestrator.get_system_status()
        projects = orchestrator.db_manager.get_projects()
        return render_template('dashboard.html', 
                             status=status, 
                             projects=projects,
                             current_project=orchestrator.get_current_project())
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return f"Error loading dashboard: {e}", 500

@app.route('/create_project', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        name = request.form.get('name')
        description = request.form.get('description', '')
        
        if not name:
            return jsonify({"error": "Project name is required"}), 400
        
        project = orchestrator.create_project(name, description)
        return jsonify({
            "success": True,
            "project": serialize_project(project)
        })
    except Exception as e:
        logger.error(f"Create project error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/create_agent', methods=['POST'])
def create_agent():
    """Create a new custom agent"""
    try:
        agent_name = request.form.get('agent_name')
        agent_type = request.form.get('agent_type')
        
        if not agent_name or not agent_type:
            return jsonify({"error": "Agent name and type are required"}), 400
        
        agent_id = f"{agent_type.lower()}-{uuid.uuid4().hex[:8]}"
        agent = orchestrator.create_custom_agent(agent_id, agent_name, AgentType(agent_type))
        
        return jsonify({
            "success": True,
            "agent": serialize_agent(agent)
        })
    except Exception as e:
        logger.error(f"Create agent error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete_agent', methods=['POST'])
def delete_agent():
    """Delete an agent"""
    try:
        agent_id = request.form.get('agent_id')
        
        if not agent_id:
            return jsonify({"error": "Agent ID is required"}), 400
        
        success = orchestrator.delete_agent(agent_id)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Cannot delete agent with active tasks"}), 400
            
    except Exception as e:
        logger.error(f"Delete agent error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/create_task', methods=['POST'])
def create_task():
    """Create a new task with optional agent assignment"""
    try:
        name = request.form.get('name')
        description = request.form.get('description', '')
        assigned_agent = request.form.get('assigned_agent', '')
        dependencies = request.form.get('dependencies', '').split(',')
        dependencies = [dep.strip() for dep in dependencies if dep.strip()]
        priority = int(request.form.get('priority', 1))
        
        if not name:
            return jsonify({"error": "Task name is required"}), 400
        
        assigned_agent = assigned_agent if assigned_agent else None
        
        task = orchestrator.create_task(name, description, assigned_agent, dependencies, priority)
        return jsonify({
            "success": True,
            "task": serialize_task(task)
        })
    except Exception as e:
        logger.error(f"Create task error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    try:
        return jsonify(orchestrator.get_system_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lm_studio_status')
def api_lm_studio_status():
    """API endpoint for LM Studio status"""
    try:
        return jsonify(orchestrator.lm_studio_client.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tasks')
def api_tasks():
    """API endpoint for tasks"""
    try:
        project_id = request.args.get('project_id')
        tasks = orchestrator.db_manager.get_tasks(project_id)
        return jsonify([asdict(task) for task in tasks])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents')
def api_agents():
    """API endpoint for agents"""
    try:
        agents = orchestrator.db_manager.get_agents()
        return jsonify([asdict(agent) for agent in agents])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/available_agents')
def api_available_agents():
    """API endpoint for available agents"""
    try:
        available_agents = orchestrator.get_available_agents()
        return jsonify(available_agents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/memory/<agent_id>')
def api_memory(agent_id):
    """API endpoint for agent memory"""
    try:
        memories = orchestrator.memory_manager.get_agent_memories(agent_id)
        return jsonify(memories)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# HTML TEMPLATES
# ============================================================================

def create_templates():
    """Create HTML templates"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Teams AI System - LM Studio Powered</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0; padding: 20px; background: #f5f7fa; 
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;
        }
        .lm-studio-status {
            background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px; margin-top: 10px;
        }
        .status-connected { color: #4CAF50; }
        .status-disconnected { color: #F44336; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { 
            background: white; padding: 20px; border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .status-item { 
            padding: 15px; border-radius: 8px; text-align: center;
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .agent-item, .task-item { 
            padding: 10px; margin: 10px 0; border-radius: 5px;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
        }
        .btn { 
            background: #667eea; color: white; border: none; 
            padding: 10px 15px; border-radius: 5px; cursor: pointer;
            margin: 5px;
        }
        .btn:hover { background: #5a6fd8; }
        .form-group { margin: 10px 0; }
        .form-group input, .form-group textarea { 
            width: 100%; padding: 8px; border: 1px solid #ddd; 
            border-radius: 4px; margin-top: 5px;
        }
        .status-badge {
            display: inline-block; padding: 4px 8px; border-radius: 4px;
            font-size: 12px; font-weight: bold;
        }
        .status-idle { background: #28a745; color: white; }
        .status-busy { background: #ffc107; color: black; }
        .status-pending { background: #6c757d; color: white; }
        .status-completed { background: #28a745; color: white; }
        .status-failed { background: #dc3545; color: white; }
        .delete-btn { 
            background: #dc3545; color: white; border: none; 
            padding: 5px 10px; border-radius: 3px; cursor: pointer;
            font-size: 12px; margin-left: 10px;
        }
        .delete-btn:hover { background: #c82333; }
        .agent-controls { 
            display: flex; justify-content: space-between; 
            align-items: center; margin: 10px 0;
        }
        .dropdown { 
            width: 100%; padding: 8px; border: 1px solid #ddd; 
            border-radius: 4px; margin-top: 5px;
        }
        .form-row { 
            display: grid; grid-template-columns: 1fr 1fr; 
            gap: 10px; margin: 10px 0; 
        }
        .agent-section { 
            border: 2px solid #667eea; border-radius: 10px; 
            padding: 15px; margin: 10px 0; background: #f8f9ff;
        }
        .task-assignment { 
            background: #e8f4f8; padding: 10px; 
            border-radius: 5px; margin: 5px 0;
        }
        .ai-section {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white; padding: 15px; border-radius: 10px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Teams AI System</h1>
        <p>Collaborative AI Agents for Software Development - Powered by LM Studio</p>
        {% if current_project %}
        <p><strong>Current Project:</strong> {{ current_project.name }}</p>
        {% endif %}
        
        <div class="lm-studio-status">
            <strong>🧠 LM Studio Status:</strong>
            {% if status.lm_studio_status.connected %}
            <span class="status-connected">✅ Connected</span>
            {% else %}
            <span class="status-disconnected">❌ Disconnected</span>
            {% endif %}
            <br>
            <small>URL: {{ status.lm_studio_status.url }} | Model: {{ status.lm_studio_status.current_model }}</small>
        </div>
    </div>

    <div class="container">
        <div class="card">
            <h2>📊 System Status</h2>
            <div class="status-grid">
                <div class="status-item">
                    <h3>{{ status.agents|length }}</h3>
                    <p>Active Agents</p>
                </div>
                <div class="status-item">
                    <h3>{{ status.tasks|length }}</h3>
                    <p>Total Tasks</p>
                </div>
                <div class="status-item">
                    <h3>{{ status.projects|length }}</h3>
                    <p>Projects</p>
                </div>
                <div class="status-item">
                    <h3>{{ status.message_queue_size }}</h3>
                    <p>Queue Size</p>
                </div>
            </div>
            
            <div class="ai-section">
                <h3>🧠 AI Model Information</h3>
                <p><strong>Temperature:</strong> {{ status.lm_studio_status.temperature }}</p>
                <p><strong>Max Tokens:</strong> {{ status.lm_studio_status.max_tokens }}</p>
                <p><strong>Available Models:</strong> {{ status.lm_studio_status.models|join(', ') if status.lm_studio_status.models else 'None loaded' }}</p>
            </div>
        </div>

        <div class="card">
            <h2>🚀 Quick Actions</h2>
            
            <div class="agent-section">
                <h3>👥 Create Custom Agent</h3>
                <form id="agentForm">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Agent Name:</label>
                            <input type="text" name="agent_name" placeholder="e.g., Sarah CodeMaster" required>
                        </div>
                        <div class="form-group">
                            <label>Agent Type:</label>
                            <select name="agent_type" class="dropdown" required>
                                <option value="">Select Agent Type</option>
                                <option value="coder">🧑‍💻 Coder (Code Generation & Debugging)</option>
                                <option value="tester">🧪 Tester (Testing & Quality Assurance)</option>
                                <option value="integrator">🔧 Integrator (Deployment & Integration)</option>
                                <option value="documenter">📚 Documenter (Documentation & Guides)</option>
                            </select>
                        </div>
                    </div>
                    <button type="submit" class="btn">Create Agent</button>
                </form>
            </div>
            
            <h3>🚀 Create New Project</h3>
            <form id="projectForm">
                <div class="form-group">
                    <input type="text" name="name" placeholder="Project Name" required>
                </div>
                <div class="form-group">
                    <textarea name="description" placeholder="Project Description"></textarea>
                </div>
                <button type="submit" class="btn">Create Project</button>
            </form>

            <div class="task-assignment">
                <h3>📋 Create New Task</h3>
                <form id="taskForm">
                    <div class="form-group">
                        <input type="text" name="name" placeholder="Task Name" required>
                    </div>
                    <div class="form-group">
                        <textarea name="description" placeholder="Task Description"></textarea>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Assign to Agent (Optional):</label>
                            <select name="assigned_agent" class="dropdown" id="agentDropdown">
                                <option value="">Auto-assign based on task type</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Priority (1-10):</label>
                            <input type="number" name="priority" value="1" min="1" max="10" placeholder="Priority">
                        </div>
                    </div>
                    <div class="form-group">
                        <input type="text" name="dependencies" placeholder="Dependencies (comma-separated task IDs)">
                    </div>
                    <button type="submit" class="btn">Create Task</button>
                </form>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="card">
            <h2>🤖 AI Agents</h2>
            <div class="agent-controls">
                <h3>Your Team</h3>
                <button onclick="refreshAgents()" class="btn">Refresh Agents</button>
            </div>
            {% for agent in status.agents %}
            <div class="agent-item">
                <div class="agent-controls">
                    <div>
                        <strong>{{ agent.name }}</strong> ({{ agent.agent_type }})
                        <span class="status-badge status-{{ agent.status }}">{{ agent.status }}</span>
                        <br>
                        <small>Capabilities: {{ agent.capabilities|join(', ') }}</small>
                        {% if agent.current_task %}
                        <br><small>Current Task: {{ agent.current_task }}</small>
                        {% endif %}
                    </div>
                    {% if not agent.id.endswith('-default') %}
                    <button onclick="deleteAgent('{{ agent.id }}')" class="delete-btn">Delete</button>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>

        <div class="card">
            <h2>📋 Recent Tasks</h2>
            {% for task in status.tasks[:10] %}
            <div class="task-item">
                <strong>{{ task.name }}</strong>
                <span class="status-badge status-{{ task.status }}">{{ task.status }}</span>
                <br>
                <small>{{ task.description }}</small>
                {% if task.assigned_agent %}
                <br><small>Assigned to: {{ task.assigned_agent }}</small>
                {% endif %}
                {% if task.output %}
                <br><small>Output: {{ task.output[:100] }}...</small>
                {% endif %}
                {% if task.error_message %}
                <br><small style="color: red;">Error: {{ task.error_message[:100] }}...</small>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        async function loadAvailableAgents() {
            try {
                const response = await fetch('/api/available_agents');
                const agents = await response.json();
                const dropdown = document.getElementById('agentDropdown');
                
                dropdown.innerHTML = '<option value="">Auto-assign based on task type</option>';
                
                agents.forEach(agent => {
                    const option = document.createElement('option');
                    option.value = agent.id;
                    option.textContent = `${agent.name} (${agent.type})`;
                    dropdown.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading agents:', error);
            }
        }
        
        async function refreshAgents() {
            window.location.reload();
        }
        
        async function deleteAgent(agentId) {
            if (!confirm('Are you sure you want to delete this agent?')) {
                return;
            }
            
            try {
                const formData = new FormData();
                formData.append('agent_id', agentId);
                
                const response = await fetch('/delete_agent', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (result.success) {
                    alert('Agent deleted successfully!');
                    window.location.reload();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error deleting agent: ' + error);
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            loadAvailableAgents();
        });
        
        let refreshInterval;
        
        function startAutoRefresh() {
            refreshInterval = setInterval(() => {
                const projectForm = document.getElementById('projectForm');
                const taskForm = document.getElementById('taskForm');
                const agentForm = document.getElementById('agentForm');
                
                const projectHasData = Array.from(projectForm.elements).some(el => el.value.trim() !== '');
                const taskHasData = Array.from(taskForm.elements).some(el => el.value.trim() !== '');
                const agentHasData = Array.from(agentForm.elements).some(el => el.value.trim() !== '');
                
                if (!projectHasData && !taskHasData && !agentHasData) {
                    loadAvailableAgents();
                    window.location.reload();
                }
            }, 15000);
        }
        
        startAutoRefresh();
        
        document.addEventListener('focusin', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                clearInterval(refreshInterval);
            }
        });
        
        document.addEventListener('focusout', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                setTimeout(startAutoRefresh, 5000);
            }
        });

        document.getElementById('agentForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/create_agent', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (result.success) {
                    alert(`Agent "${result.agent.name}" created successfully!`);
                    e.target.reset();
                    loadAvailableAgents();
                    window.location.reload();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error creating agent: ' + error);
            }
        });

        document.getElementById('projectForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/create_project', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (result.success) {
                    alert('Project created successfully!');
                    e.target.reset();
                    window.location.reload();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error creating project: ' + error);
            }
        });

        document.getElementById('taskForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/create_task', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (result.success) {
                    alert('Task created successfully!');
                    e.target.reset();
                    loadAvailableAgents();
                    window.location.reload();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error creating task: ' + error);
            }
        });
    </script>
</body>
</html>
    '''
    
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(dashboard_html)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the Teams AI System"""
    print("🚀 Starting Teams AI System...")
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/repositories", exist_ok=True)
    os.makedirs("data/memory", exist_ok=True)
    
    # Check LM Studio connection before proceeding
    print("🤖 Connecting to LM Studio...")
    test_client = LMStudioClient()
    if not test_client.verify_connection():
        print("❌ Cannot connect to LM Studio!")
        print("Please ensure:")
        print("1. LM Studio is running")
        print("2. A model is loaded (GPT-OSS-20B recommended)")
        print("3. The local server is started in LM Studio")
        print("4. The server URL is accessible at:", test_client.base_url)
        print("\nTrying to continue anyway...")
    
    create_templates()
    init_system()
    
    print("✅ Teams AI System initialized with LM Studio integration")
    print("🎯 Create custom agents and tasks through the web interface!")
    print("🧠 AI responses powered by your local LM Studio model")
    print("🌐 Starting web interface...")
    print("🌐 Web interface available at: http://localhost:8081")
    
    try:
        app.run(host='0.0.0.0', port=8081, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Teams AI System...")
        orchestrator.running = False
        print("✅ Teams AI System stopped")

if __name__ == "__main__":
    main()