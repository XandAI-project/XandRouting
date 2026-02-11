# Contributing to Multi-LLM Server

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Style Guides](#style-guides)

---

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

---

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Git** installed and configured
- **Docker** and **Docker Compose** installed
- **Python 3.10+** for local development
- **NVIDIA GPU** with CUDA (optional, for testing GPU features)
- **NVIDIA Container Toolkit** (for GPU testing)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/multi-llm-server.git
cd multi-llm-server
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/multi-llm-server.git
```

4. Create a new branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

**Branch Naming Conventions:**

- `feature/short-description` - New features
- `bugfix/short-description` - Bug fixes
- `hotfix/short-description` - Urgent production fixes
- `chore/short-description` - Maintenance tasks
- `docs/short-description` - Documentation updates

---

## 🛠️ Development Setup

### Local Development (Docker)

The easiest way to develop is using Docker Compose:

```bash
# Build and start the container
docker compose up --build

# View logs
docker compose logs -f gateway

# Rebuild after changes
docker compose build --no-cache gateway
docker compose up
```

### Local Development (Python Virtual Environment)

For faster iteration during development:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
cd gateway
pip install -r requirements.txt

# Run the server
python main.py
```

The server will be available at `http://localhost:8080`.

### Environment Variables

Create a `.env` file in the project root (never commit this file):

```bash
# Model cache configuration
DEFAULT_TTL=300
CLEANUP_INTERVAL=30

# Server configuration
PORT=8080
HOST=0.0.0.0

# Optional: HuggingFace token for private models
HF_TOKEN=your_token_here
```

---

## 📐 Code Quality Standards

We follow clean code principles and Python best practices.

### Clean Code Principles

- **Single Responsibility Principle** - Each class/function should do one thing well
- **DRY (Don't Repeat Yourself)** - Avoid code duplication
- **Meaningful Names** - Use descriptive variable and function names
- **Small Functions** - Keep functions focused and concise
- **Proper Error Handling** - Handle errors gracefully with informative messages

### Code Structure

Follow clean architecture patterns:

- **Separation of Concerns** - Separate API routes, business logic, and data access
- **Dependency Inversion** - Depend on abstractions, not concrete implementations
- **Layer Isolation** - Keep layers independent and testable

### Python Style Guide

We follow **PEP 8** style guidelines:

- 4 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Use snake_case for functions and variables
- Use PascalCase for classes
- Use UPPER_CASE for constants
- Add docstrings to all public functions and classes

**Example:**

```python
def calculate_cache_key(model_path: str, backend: str, device: str) -> str:
    """
    Generate a unique cache key for a model configuration.
    
    Args:
        model_path: Path to the model directory
        backend: Backend engine (vllm, transformers, llamacpp)
        device: Device type (cuda, cpu)
        
    Returns:
        Unique cache key string
    """
    return f"{model_path}_{backend}_{device}"
```

### Type Hints

Always use type hints for function parameters and return values:

```python
from typing import List, Dict, Optional

def process_messages(
    messages: List[Dict[str, str]],
    max_tokens: int = 512
) -> Optional[str]:
    """Process chat messages and return formatted prompt"""
    if not messages:
        return None
    
    return format_prompt(messages, max_tokens)
```

### Linting and Formatting

We use the following tools (configuration provided in project root):

- **Black** - Code formatter
- **isort** - Import sorting
- **flake8** - Linting
- **mypy** - Type checking

Run before committing:

```bash
# Format code
black gateway/

# Sort imports
isort gateway/

# Check linting
flake8 gateway/

# Check types
mypy gateway/
```

---

## 🔨 Making Changes

### Before You Start

1. **Check existing issues** - Ensure your change isn't already being worked on
2. **Open an issue** - Discuss significant changes before starting work
3. **Read existing code** - Understand patterns and conventions used in the project
4. **Start small** - Begin with a focused change that solves one problem

### Error Handling Procedure

When encountering errors during development:

1. **Log Everything** - Capture and log all relevant payloads, request data, and context
2. **Review Business Logic** - Verify implementation matches requirements
3. **Analyze Error Output** - Examine stack traces and error messages
4. **Ask for Guidance** - If unable to resolve, open an issue with:
   - Complete error output
   - What you attempted
   - Relevant code context
   - Suggested solutions

**Never guess** - When uncertain about requirements or implementation, ask first.

### Understanding the Codebase

Before making changes:

- Read and analyze existing code extensively
- Understand patterns, conventions, and architecture
- Match existing code style and structure
- Identify and reuse existing utilities

**Key Files to Review:**

- `gateway/main.py` - API endpoints and request handling
- `gateway/inference.py` - Inference engine implementation
- `gateway/model_cache.py` - Cache management and TTL logic
- `gateway/model_loader.py` - Backend-specific model loading
- `gateway/download_manager.py` - Download orchestration

---

## 🧪 Testing

### API Endpoint Testing

**All API endpoints must be tested before considering implementation complete.**

#### Manual Testing with curl

Test the chat completions endpoint:

```bash
# Non-streaming
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/test-model",
    "backend": "llamacpp",
    "device": "cuda",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Streaming
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/test-model",
    "backend": "llamacpp",
    "device": "cuda",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

Test model management:

```bash
# List loaded models
curl http://localhost:8080/v1/models/loaded

# Get cache stats
curl http://localhost:8080/v1/models/stats

# List inventory
curl http://localhost:8080/v1/models/inventory

# Unload all models
curl -X POST http://localhost:8080/v1/models/unload-all
```

Test download endpoints:

```bash
# Start download
curl -X POST http://localhost:8080/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://huggingface.co/org/model",
    "quantization": "Q4_K_M"
  }'

# Check status
curl http://localhost:8080/v1/models/download/download_123456

# List all downloads
curl http://localhost:8080/v1/models/download
```

#### Testing Checklist

For each endpoint change, verify:

- ✅ Request validates correctly (400 for invalid requests)
- ✅ Response matches documented schema
- ✅ Error cases return appropriate status codes
- ✅ Edge conditions are handled (empty arrays, null values, etc.)
- ✅ Streaming works correctly (if applicable)
- ✅ Logs contain useful information for debugging

### Integration Testing

Test complete workflows:

1. **Download → Load → Infer** - Full end-to-end flow
2. **Cache behavior** - Verify TTL and automatic unloading
3. **Multiple backends** - Test vLLM, Transformers, llama.cpp
4. **Error scenarios** - Invalid models, OOM, network failures

### Performance Testing

For performance-critical changes:

- Test with realistic model sizes
- Measure inference latency
- Monitor memory usage
- Check cache hit rates

---

## 🔄 Pull Request Process

### Before Submitting

1. **Update from upstream** - Rebase on latest main branch

```bash
git fetch upstream
git rebase upstream/main
```

2. **Run linters** - Fix all linting errors

```bash
black gateway/
isort gateway/
flake8 gateway/
```

3. **Test thoroughly** - Verify all affected endpoints work
4. **Update documentation** - Update README.md if adding features
5. **Write clear commits** - Follow commit message guidelines

### Commit Messages

Follow these guidelines for commit messages:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line: concise summary (50 characters or less)
- Second line: blank
- Additional lines: detailed explanation (wrap at 72 characters)
- Focus on **why**, not **what** (code shows what changed)

**Good Examples:**

```
Add streaming support for Transformers backend

Implements TextIteratorStreamer for real-time token streaming.
Resolves issue where Transformers backend only supported complete
responses, making it inconsistent with vLLM and llama.cpp.

Fixes #42
```

```
Fix memory leak in model cache cleanup

The cleanup thread wasn't properly releasing model references,
causing models to stay in memory after TTL expiration.

Added explicit model.unload() calls and verified with memory profiling.
```

### Creating a Pull Request

1. **Push your branch** to your fork:

```bash
git push origin feature/your-feature-name
```

2. **Open a Pull Request** on GitHub from your fork to the upstream repository

3. **Fill out the PR template** completely (see `.github/PULL_REQUEST_TEMPLATE.md`)

### PR Requirements

Your pull request must include:

- **Clear title** - Descriptive summary of changes
- **Description** - What changed and why
- **Testing** - How to test the changes
- **Screenshots** - If applicable (UI changes, error messages)
- **Breaking changes** - Document any breaking changes
- **Issue reference** - Link to related issues

**PR Template Structure:**

```markdown
## Summary
Brief description of changes (3-5 sentences)

## Changes Made
- Bullet list of specific changes
- What was added/modified/removed

## Testing
How to test these changes:
1. Step-by-step instructions
2. Expected results
3. Edge cases covered

## Breaking Changes
- Any breaking changes and migration steps
- Or "None"

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] Tests pass locally
- [ ] Dependent changes merged

Fixes #(issue_number)
```

### Review Process

1. **Maintainer review** - A maintainer will review your PR
2. **Address feedback** - Make requested changes
3. **Update PR** - Push additional commits (don't force-push during review)
4. **Approval** - PR approved by maintainer(s)
5. **Merge** - Maintainer merges your PR

---

## 📝 Issue Guidelines

### Before Opening an Issue

- Search existing issues (open and closed)
- Check if it's already fixed in main branch
- Verify it's not a configuration issue

### Bug Reports

Use the bug report template (`.github/ISSUE_TEMPLATE/bug_report.md`).

Include:

- **Environment** - OS, Docker version, GPU info
- **Steps to reproduce** - Exact commands/requests to reproduce
- **Expected behavior** - What should happen
- **Actual behavior** - What actually happens
- **Logs** - Relevant error messages and stack traces
- **Configuration** - Relevant environment variables and settings

### Feature Requests

Use the feature request template (`.github/ISSUE_TEMPLATE/feature_request.md`).

Include:

- **Problem statement** - What problem does this solve?
- **Proposed solution** - How should it work?
- **Alternatives considered** - Other approaches you've thought about
- **Use case** - Real-world scenario where this would be useful

---

## 📚 Style Guides

### Documentation

- Write in clear, concise English
- Use code blocks for commands and code examples
- Include practical examples
- Update README.md for user-facing changes
- Update ARCHITECTURE.md for internal changes

### API Documentation

For new endpoints, document:

- **Request schema** - Parameters, types, validation rules
- **Response schema** - Success and error cases
- **Example payloads** - Real-world request/response examples
- **Error codes** - Possible error codes and meanings

**Example:**

```python
@app.post("/v1/models/example")
async def example_endpoint(request: ExampleRequest):
    """
    Example endpoint that demonstrates proper documentation.
    
    Request Body:
    - param1 (str, required): Description of parameter
    - param2 (int, optional): Description with default value
    
    Response:
    - 200: Success response with ExampleResponse schema
    - 400: Invalid request (missing required fields)
    - 404: Resource not found
    - 500: Internal server error
    
    Example:
        POST /v1/models/example
        {
            "param1": "value",
            "param2": 42
        }
        
        Response:
        {
            "result": "success",
            "data": {...}
        }
    """
    # Implementation
```

### Logging

Use structured logging with appropriate levels:

```python
import logging

logger = logging.getLogger(__name__)

# Debug - Detailed diagnostic information
logger.debug(f"Cache key generated: {cache_key}")

# Info - General informational messages
logger.info(f"Model loaded successfully: {model_path}")

# Warning - Something unexpected but not critical
logger.warning(f"High memory usage detected: {usage}%")

# Error - Error condition but application can continue
logger.error(f"Failed to download model: {error}", exc_info=True)

# Critical - Critical error requiring immediate attention
logger.critical(f"Out of memory, cannot load model: {model_path}")
```

---

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

### High Priority

- **Additional backends** - Support for more inference engines
- **Performance optimizations** - Faster model loading, inference
- **Better error messages** - More helpful error reporting
- **Testing infrastructure** - Unit tests, integration tests

### Medium Priority

- **Monitoring/metrics** - Prometheus metrics, health checks
- **Model registry** - Database-backed model catalog
- **Authentication** - API key support, user management
- **Request queuing** - Handle concurrent requests better

### Documentation

- **Examples** - More language examples (Java, Go, etc.)
- **Tutorials** - Step-by-step guides for common use cases
- **Videos** - Video demonstrations
- **Blog posts** - Technical deep dives

---

## 📞 Questions?

- **GitHub Issues** - Technical questions about the codebase
- **GitHub Discussions** - General questions and community discussion
- **Documentation** - Check README.md and ARCHITECTURE.md first

---

## 🙏 Thank You!

Every contribution, no matter how small, helps improve the project. We appreciate your time and effort!

**Happy coding! 🚀**
