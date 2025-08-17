# Teams AI System - Multi-Agent Software Development Framework
# A fully working prototype of collaborative AI agents for software development

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
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
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
                f.write(f"# {project_name}\n\nGenerated by Teams AI System\n")
            
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
    
    def create_branch(self, repo_path: str, branch_name: str):
        """Create and switch to a new branch"""
        try:
            subprocess.run(['git', 'checkout', '-b', branch_name], 
                         cwd=repo_path, check=True, capture_output=True)
            logger.info(f"Created branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Branch creation failed: {e}")
    
    def merge_branch(self, repo_path: str, branch_name: str):
        """Merge a branch into main"""
        try:
            subprocess.run(['git', 'checkout', 'main'], cwd=repo_path, check=True)
            subprocess.run(['git', 'merge', branch_name], cwd=repo_path, check=True)
            logger.info(f"Merged branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Merge failed: {e}")

# ============================================================================
# AI AGENT BASE CLASS
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all AI agents"""
    
    def __init__(self, agent_id: str, name: str, agent_type: AgentType, 
                 orchestrator, memory_manager: MemoryManager):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.orchestrator = orchestrator
        self.memory_manager = memory_manager
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
    def execute_task(self, task: Task) -> str:
        """Execute a specific task"""
        pass
    
    def simulate_llm_call(self, prompt: str, context: str = "") -> str:
        """Simulate LLM API call (replace with actual LLM integration)"""
        # In a real implementation, this would call OpenAI, Anthropic, etc.
        responses = {
            "code_generation": f"# Generated code for: {prompt}\nprint('Hello from AI!')",
            "test_generation": f"# Test for: {prompt}\nassert True",
            "documentation": f"# Documentation for: {prompt}\nThis module does amazing things.",
            "integration": f"# Integration completed for: {prompt}"
        }
        
        # Simple keyword matching for demo
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        
        return f"# AI Response to: {prompt}\n# Context: {context}"
    
    def update_status(self, status: str):
        """Update agent status"""
        self.status = status
        self.orchestrator.update_agent_status(self.agent_id, status)
    
    def send_message(self, receiver_id: str, message_type: MessageType, content: Dict):
        """Send message to another agent or orchestrator"""
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now()
        )
        self.orchestrator.route_message(message)
    
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
    
    def execute_task(self, task: Task) -> str:
        self.update_status("coding")
        self.current_task = task.id
        
        # Simulate code generation
        prompt = f"Generate code for: {task.description}"
        context = self.get_task_context(task)
        
        code = self.simulate_llm_call(prompt, context)
        
        # Write code to file
        file_path = self.write_code_to_file(task, code)
        
        self.log_activity(f"Generated code for task: {task.name}")
        self.update_status("idle")
        
        return f"Code generated and saved to: {file_path}"
    
    def get_task_context(self, task: Task) -> str:
        """Get relevant context from memory and dependencies"""
        memories = self.memory_manager.get_agent_memories(self.agent_id)
        context = "Previous activities:\n"
        for memory in memories[-3:]:  # Last 3 activities
            context += f"- {memory['content']}\n"
        return context
    
    def write_code_to_file(self, task: Task, code: str) -> str:
        """Write generated code to project file"""
        # Get project repository path
        project = self.orchestrator.get_current_project()
        if not project:
            return "No active project"
        
        file_name = f"{task.name.lower().replace(' ', '_')}.py"
        file_path = Path(project.repository_path) / "src" / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(code)
        
        return str(file_path)

class TesterAgent(BaseAgent):
    """AI Agent specialized in testing"""
    
    def get_capabilities(self) -> List[str]:
        return ["unit_testing", "integration_testing", "test_coverage", "test_automation"]
    
    def execute_task(self, task: Task) -> str:
        self.update_status("testing")
        self.current_task = task.id
        
        # Generate test code
        prompt = f"Generate tests for: {task.description}"
        context = self.get_testing_context(task)
        
        test_code = self.simulate_llm_call(f"test_generation {prompt}", context)
        
        # Write test file
        test_file_path = self.write_test_file(task, test_code)
        
        # Simulate running tests
        test_results = self.run_tests(test_file_path)
        
        self.log_activity(f"Created and ran tests for task: {task.name}")
        self.update_status("idle")
        
        return f"Tests created: {test_file_path}, Results: {test_results}"
    
    def get_testing_context(self, task: Task) -> str:
        """Get context for test generation"""
        return f"Task dependencies: {task.dependencies}"
    
    def write_test_file(self, task: Task, test_code: str) -> str:
        """Write test code to file"""
        project = self.orchestrator.get_current_project()
        if not project:
            return "No active project"
        
        test_file_name = f"test_{task.name.lower().replace(' ', '_')}.py"
        test_file_path = Path(project.repository_path) / "tests" / test_file_name
        test_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_file_path, 'w') as f:
            f.write(test_code)
        
        return str(test_file_path)
    
    def run_tests(self, test_file_path: str) -> str:
        """Simulate running tests"""
        return "All tests passed (simulated)"

class IntegratorAgent(BaseAgent):
    """AI Agent specialized in integration and deployment"""
    
    def get_capabilities(self) -> List[str]:
        return ["code_integration", "deployment", "ci_cd", "merge_management"]
    
    def execute_task(self, task: Task) -> str:
        self.update_status("integrating")
        self.current_task = task.id
        
        project = self.orchestrator.get_current_project()
        if not project:
            return "No active project"
        
        # Simulate integration process
        self.integrate_code_changes(project, task)
        
        self.log_activity(f"Integrated changes for task: {task.name}")
        self.update_status("idle")
        
        return f"Integration completed for task: {task.name}"
    
    def integrate_code_changes(self, project: Project, task: Task):
        """Integrate code changes into main branch"""
        vc_manager = self.orchestrator.version_control_manager
        
        # Commit current changes
        commit_message = f"Implement {task.name}: {task.description}"
        vc_manager.commit_changes(project.repository_path, commit_message)

class DocumenterAgent(BaseAgent):
    """AI Agent specialized in documentation"""
    
    def get_capabilities(self) -> List[str]:
        return ["api_documentation", "code_comments", "readme_generation", "user_guides"]
    
    def execute_task(self, task: Task) -> str:
        self.update_status("documenting")
        self.current_task = task.id
        
        # Generate documentation
        prompt = f"Generate documentation for: {task.description}"
        context = self.get_documentation_context(task)
        
        documentation = self.simulate_llm_call(f"documentation {prompt}", context)
        
        # Write documentation file
        doc_file_path = self.write_documentation_file(task, documentation)
        
        self.log_activity(f"Created documentation for task: {task.name}")
        self.update_status("idle")
        
        return f"Documentation created: {doc_file_path}"
    
    def get_documentation_context(self, task: Task) -> str:
        """Get context for documentation generation"""
        return f"Task type: {task.name}, Dependencies: {task.dependencies}"
    
    def write_documentation_file(self, task: Task, documentation: str) -> str:
        """Write documentation to file"""
        project = self.orchestrator.get_current_project()
        if not project:
            return "No active project"
        
        doc_file_name = f"{task.name.lower().replace(' ', '_')}_docs.md"
        doc_file_path = Path(project.repository_path) / "docs" / doc_file_name
        doc_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(doc_file_path, 'w') as f:
            f.write(documentation)
        
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
        
        # Start orchestrator thread
        self.orchestrator_thread = threading.Thread(target=self.run_orchestrator)
        self.orchestrator_thread.daemon = True
    
    def start(self):
        """Start the orchestrator"""
        self.running = True
        self.orchestrator_thread.start()
        logger.info("TaskOrchestrator started")
    
    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        logger.info("TaskOrchestrator stopped")
    
    def register_agent(self, agent: Agent):
        """Register a new agent"""
        self.agent_objects[agent.id] = agent
        self.db_manager.save_agent(agent)
        logger.info(f"Registered agent: {agent.name} ({agent.agent_type.value})")
    
    def add_agent_instance(self, agent: BaseAgent):
        """Add agent instance for task execution"""
        self.agents[agent.agent_id] = agent
    
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
    
    def create_task(self, name: str, description: str, dependencies: List[str] = None, 
                   priority: int = 1) -> Task:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            description=description,
            status=TaskStatus.PENDING,
            assigned_agent=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            dependencies=dependencies or [],
            priority=priority
        )
        
        self.tasks[task_id] = task
        if self.current_project_id:
            self.db_manager.save_task(task, self.current_project_id)
        
        logger.info(f"Created task: {name}")
        return task
    
    def assign_task(self, task_id: str, agent_type: AgentType) -> bool:
        """Assign a task to an appropriate agent"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # Find available agent of the requested type
        available_agent = None
        for agent_id, agent in self.agents.items():
            if (agent.agent_type == agent_type and 
                agent.status == "idle" and 
                agent.current_task is None):
                available_agent = agent
                break
        
        if not available_agent:
            logger.warning(f"No available {agent_type.value} agent for task {task.name}")
            return False
        
        # Assign task
        task.assigned_agent = available_agent.agent_id
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()
        
        # Update database
        if self.current_project_id:
            self.db_manager.save_task(task, self.current_project_id)
        
        # Send task assignment message
        message = Message(
            id=str(uuid.uuid4()),
            sender_id="orchestrator",
            receiver_id=available_agent.agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            content={"task": asdict(task)},
            timestamp=datetime.now()
        )
        
        self.message_queue.append(message)
        logger.info(f"Assigned task {task.name} to {available_agent.name}")
        return True
    
    def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        if agent_id in self.agent_objects:
            self.agent_objects[agent_id].status = status
            self.db_manager.save_agent(self.agent_objects[agent_id])
    
    def route_message(self, message: Message):
        """Route message between agents"""
        self.message_queue.append(message)
    
    def complete_task(self, task_id: str, output: str, agent_id: str):
        """Mark task as completed"""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.output = output
            task.updated_at = datetime.now()
            
            if self.current_project_id:
                self.db_manager.save_task(task, self.current_project_id)
            
            # Update agent
            if agent_id in self.agents:
                self.agents[agent_id].current_task = None
                self.agents[agent_id].update_status("idle")
            
            logger.info(f"Task completed: {task.name}")
            
            # Check for dependent tasks
            self.check_dependency_completion(task_id)
    
    def check_dependency_completion(self, completed_task_id: str):
        """Check if any pending tasks can now be started"""
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                completed_task_id in task.dependencies):
                
                # Check if all dependencies are completed
                all_deps_completed = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                    if dep_id in self.tasks
                )
                
                if all_deps_completed:
                    # Auto-assign based on task type
                    agent_type = self.determine_agent_type_for_task(task)
                    self.assign_task(task.id, agent_type)
    
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
            return AgentType.CODER  # Default
    
    def run_orchestrator(self):
        """Main orchestrator loop"""
        while self.running:
            try:
                # Process message queue
                if self.message_queue:
                    message = self.message_queue.pop(0)
                    self.process_message(message)
                
                # Check for auto-assignable tasks
                self.auto_assign_ready_tasks()
                
                time.sleep(1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Orchestrator error: {e}")
                time.sleep(5)
    
    def process_message(self, message: Message):
        """Process a message"""
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            # Execute task on agent
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
                # Check if dependencies are satisfied
                deps_satisfied = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                    if dep_id in self.tasks
                )
                
                if deps_satisfied or not task.dependencies:
                    agent_type = self.determine_agent_type_for_task(task)
                    self.assign_task(task.id, agent_type)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "agents": [asdict(agent) for agent in self.agent_objects.values()],
            "tasks": [asdict(task) for task in self.tasks.values()],
            "projects": [asdict(project) for project in self.projects.values()],
            "current_project": self.current_project_id,
            "message_queue_size": len(self.message_queue)
        }

# ============================================================================
# WEB INTERFACE
# ============================================================================

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "teams-ai-secret-key"

# Global orchestrator instance
orchestrator = TaskOrchestrator()

def init_system():
    """Initialize the Teams AI system"""
    # Create AI agents
    coder = CoderAgent("coder-001", "Alice Coder", AgentType.CODER, 
                      orchestrator, orchestrator.memory_manager)
    tester = TesterAgent("tester-001", "Bob Tester", AgentType.TESTER, 
                        orchestrator, orchestrator.memory_manager)
    integrator = IntegratorAgent("integrator-001", "Charlie Integrator", 
                                AgentType.INTEGRATOR, orchestrator, orchestrator.memory_manager)
    documenter = DocumenterAgent("documenter-001", "Diana Documenter", 
                                AgentType.DOCUMENTER, orchestrator, orchestrator.memory_manager)
    
    # Add agent instances to orchestrator
    orchestrator.add_agent_instance(coder)
    orchestrator.add_agent_instance(tester)
    orchestrator.add_agent_instance(integrator)
    orchestrator.add_agent_instance(documenter)
    
    # Start orchestrator
    orchestrator.start()
    
    logger.info("Teams AI System initialized")

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
            "project": asdict(project)
        })
    except Exception as e:
        logger.error(f"Create project error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/create_task', methods=['POST'])
def create_task():
    """Create a new task"""
    try:
        name = request.form.get('name')
        description = request.form.get('description', '')
        dependencies = request.form.get('dependencies', '').split(',')
        dependencies = [dep.strip() for dep in dependencies if dep.strip()]
        priority = int(request.form.get('priority', 1))
        
        if not name:
            return jsonify({"error": "Task name is required"}), 400
        
        task = orchestrator.create_task(name, description, dependencies, priority)
        return jsonify({
            "success": True,
            "task": asdict(task)
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

# Create templates directory and files
def create_templates():
    """Create HTML templates"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Dashboard template
    dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Teams AI System</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0; padding: 20px; background: #f5f7fa; 
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;
        }
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
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Teams AI System</h1>
        <p>Collaborative AI Agents for Software Development</p>
        {% if current_project %}
        <p><strong>Current Project:</strong> {{ current_project.name }}</p>
        {% endif %}
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
        </div>

        <div class="card">
            <h2>🚀 Quick Actions</h2>
            
            <h3>Create New Project</h3>
            <form id="projectForm">
                <div class="form-group">
                    <input type="text" name="name" placeholder="Project Name" required>
                </div>
                <div class="form-group">
                    <textarea name="description" placeholder="Project Description"></textarea>
                </div>
                <button type="submit" class="btn">Create Project</button>
            </form>

            <h3>Create New Task</h3>
            <form id="taskForm">
                <div class="form-group">
                    <input type="text" name="name" placeholder="Task Name" required>
                </div>
                <div class="form-group">
                    <textarea name="description" placeholder="Task Description"></textarea>
                </div>
                <div class="form-group">
                    <input type="text" name="dependencies" placeholder="Dependencies (comma-separated task IDs)">
                </div>
                <div class="form-group">
                    <input type="number" name="priority" value="1" min="1" max="10" placeholder="Priority">
                </div>
                <button type="submit" class="btn">Create Task</button>
            </form>
        </div>
    </div>

    <div class="container">
        <div class="card">
            <h2>🤖 AI Agents</h2>
            {% for agent in status.agents %}
            <div class="agent-item">
                <strong>{{ agent.name }}</strong> ({{ agent.agent_type }})
                <span class="status-badge status-{{ agent.status }}">{{ agent.status }}</span>
                <br>
                <small>Capabilities: {{ agent.capabilities|join(', ') }}</small>
                {% if agent.current_task %}
                <br><small>Current Task: {{ agent.current_task }}</small>
                {% endif %}
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
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        // Auto-refresh every 5 seconds
        setInterval(() => {
            window.location.reload();
        }, 5000);

        // Handle project form
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

        // Handle task form
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
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/repositories", exist_ok=True)
    os.makedirs("data/memory", exist_ok=True)
    
    # Create templates
    create_templates()
    
    # Initialize system
    init_system()
    
    # Example: Create a sample project and tasks
    sample_project = orchestrator.create_project(
        "Web Dashboard",
        "Build a responsive web dashboard with user authentication"
    )
    
    # Create interdependent tasks
    task1 = orchestrator.create_task(
        "User Authentication Module",
        "Implement user login, registration, and session management",
        priority=1
    )
    
    task2 = orchestrator.create_task(
        "Dashboard UI Components",
        "Create reusable UI components for the dashboard",
        dependencies=[task1.id],
        priority=2
    )
    
    task3 = orchestrator.create_task(
        "Unit Tests for Auth",
        "Write comprehensive tests for authentication module",
        dependencies=[task1.id],
        priority=1
    )
    
    task4 = orchestrator.create_task(
        "API Documentation",
        "Generate documentation for all API endpoints",
        dependencies=[task1.id, task2.id],
        priority=3
    )
    
    task5 = orchestrator.create_task(
        "Deploy to Production",
        "Deploy the application to production environment",
        dependencies=[task1.id, task2.id, task3.id, task4.id],
        priority=1
    )
    
    print("✅ Sample project and tasks created")
    print("🌐 Starting web interface...")
    
    # Start Flask app (using port 8080 to avoid conflicts with macOS AirPlay)
    print("🌐 Web interface available at: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == "__main__":
    main()