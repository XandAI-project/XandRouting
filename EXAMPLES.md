# Examples

Comprehensive examples for using the Multi-LLM Server API with different programming languages and scenarios.

---

## Table of Contents

- [curl Examples](#curl-examples)
- [Python Examples](#python-examples)
- [Node.js Examples](#nodejs-examples)
- [Common Patterns](#common-patterns)
- [Advanced Usage](#advanced-usage)
- [Error Handling](#error-handling)
- [Production Deployment](#production-deployment)

---

## curl Examples

### Basic Chat Completion

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen2.5-coder-7b-instruct-gguf",
    "backend": "llamacpp",
    "device": "cuda",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Docker?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Streaming Response

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen2.5-coder-7b-instruct-gguf",
    "backend": "llamacpp",
    "device": "cuda",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Explain neural networks in simple terms"}
    ],
    "max_tokens": 300
  }'
```

### Download a Model

```bash
# Start download
curl -X POST http://localhost:8080/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "quantization": "Q4_K_M"
  }'

# Response: {"job_id": "download_1707689234567", "status": "downloading", ...}

# Check download status
curl http://localhost:8080/v1/models/download/download_1707689234567

# List all downloads
curl http://localhost:8080/v1/models/download
```

### Model Management

```bash
# List loaded models
curl http://localhost:8080/v1/models/loaded

# Get cache statistics
curl http://localhost:8080/v1/models/stats

# List available models
curl http://localhost:8080/v1/models/inventory

# Unload specific model
curl -X POST http://localhost:8080/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen2.5-coder-7b-instruct-gguf",
    "backend": "llamacpp",
    "device": "cuda"
  }'

# Unload all models
curl -X POST http://localhost:8080/v1/models/unload-all
```

---

## Python Examples

### Basic Client

```python
import requests
import json

class LLMClient:
    """Simple client for Multi-LLM Server"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    def chat(
        self,
        model: str,
        messages: list,
        backend: str = "llamacpp",
        device: str = "cuda",
        stream: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Send chat completion request
        
        Args:
            model: Model path (e.g., "/models/qwen2.5-coder-7b-instruct-gguf")
            messages: List of message dicts with 'role' and 'content'
            backend: Backend engine (vllm, transformers, llamacpp)
            device: Device (cuda, cpu)
            stream: Enable streaming
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Response dict or generator for streaming
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "backend": backend,
            "device": device,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        if stream:
            return self._stream_response(url, payload)
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    def _stream_response(self, url: str, payload: dict):
        """Handle streaming response"""
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        
                        if data == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            continue


# Example usage
if __name__ == "__main__":
    client = LLMClient()
    
    # Non-streaming example
    response = client.chat(
        model="/models/qwen2.5-coder-7b-instruct-gguf",
        messages=[
            {"role": "system", "content": "You are a Python expert."},
            {"role": "user", "content": "Write a function to check if a string is a palindrome"}
        ],
        backend="llamacpp",
        device="cuda",
        max_tokens=300
    )
    
    print("Response:")
    print(response['choices'][0]['message']['content'])
    print(f"\nTokens used: {response['usage']['total_tokens']}")
```

### Streaming Example

```python
def streaming_example():
    """Example of streaming chat completion"""
    client = LLMClient()
    
    print("Streaming response:\n")
    
    for chunk in client.chat(
        model="/models/qwen2.5-coder-7b-instruct-gguf",
        messages=[{"role": "user", "content": "Explain recursion with an example"}],
        backend="llamacpp",
        device="cuda",
        stream=True,
        max_tokens=400
    ):
        # Extract content from chunk
        if 'choices' in chunk and len(chunk['choices']) > 0:
            delta = chunk['choices'][0].get('delta', {})
            content = delta.get('content', '')
            
            if content:
                print(content, end='', flush=True)
    
    print("\n\nStreaming complete!")


if __name__ == "__main__":
    streaming_example()
```

### Multi-Model Workflow

```python
import time

def multi_model_example():
    """Example using multiple models and backends"""
    client = LLMClient()
    
    models = [
        {
            "name": "Code Generator",
            "model": "/models/qwen2.5-coder-7b-instruct-gguf",
            "backend": "llamacpp",
            "device": "cuda"
        },
        {
            "name": "General Assistant",
            "model": "/models/llama-3-8b",
            "backend": "transformers",
            "device": "cpu"
        }
    ]
    
    for config in models:
        print(f"\nUsing {config['name']}:")
        print("-" * 50)
        
        start_time = time.time()
        
        response = client.chat(
            model=config['model'],
            backend=config['backend'],
            device=config['device'],
            messages=[
                {"role": "user", "content": "What is your purpose?"}
            ],
            max_tokens=100
        )
        
        elapsed = time.time() - start_time
        
        print(response['choices'][0]['message']['content'])
        print(f"\nTime: {elapsed:.2f}s")
        print(f"Tokens: {response['usage']['total_tokens']}")


if __name__ == "__main__":
    multi_model_example()
```

### Model Download and Wait

```python
import time

def download_and_wait(url: str, quantization: str = None):
    """Download model and wait for completion"""
    client = LLMClient()
    
    # Start download
    response = requests.post(
        f"{client.base_url}/v1/models/download",
        json={
            "url": url,
            "quantization": quantization
        }
    )
    response.raise_for_status()
    job = response.json()
    job_id = job['job_id']
    
    print(f"Download started: {job_id}")
    print(f"Destination: {job['destination']}")
    
    # Poll for completion
    while True:
        response = requests.get(f"{client.base_url}/v1/models/download/{job_id}")
        response.raise_for_status()
        job = response.json()
        
        status = job['status']
        print(f"Status: {status} - {job.get('message', '')}")
        
        if status == 'completed':
            print(f"\nDownload complete!")
            print(f"Files downloaded: {job.get('files_downloaded', 0)}")
            print(f"Duration: {job.get('duration_seconds', 0)}s")
            return job['destination']
        
        elif status == 'failed':
            print(f"\nDownload failed: {job.get('error', 'Unknown error')}")
            return None
        
        elif status == 'cancelled':
            print("\nDownload was cancelled")
            return None
        
        time.sleep(5)  # Poll every 5 seconds


# Example usage
if __name__ == "__main__":
    model_path = download_and_wait(
        url="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        quantization="Q4_K_M"
    )
    
    if model_path:
        # Use the downloaded model
        client = LLMClient()
        response = client.chat(
            model=model_path,
            messages=[{"role": "user", "content": "Hello!"}],
            backend="llamacpp",
            device="cuda"
        )
        print(response['choices'][0]['message']['content'])
```

### Async Python Client

```python
import aiohttp
import asyncio
import json

class AsyncLLMClient:
    """Async client for Multi-LLM Server"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    async def chat(
        self,
        model: str,
        messages: list,
        backend: str = "llamacpp",
        device: str = "cuda",
        stream: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ):
        """Async chat completion"""
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "backend": backend,
            "device": device,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            if stream:
                async for chunk in self._stream_response(session, url, payload):
                    yield chunk
            else:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    return await response.json()
    
    async def _stream_response(self, session, url, payload):
        """Handle async streaming"""
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line.startswith('data: '):
                    data = line[6:]
                    
                    if data == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue


async def async_example():
    """Example using async client"""
    client = AsyncLLMClient()
    
    # Multiple concurrent requests
    tasks = [
        client.chat(
            model="/models/qwen2.5-coder-7b-instruct-gguf",
            messages=[{"role": "user", "content": f"What is {topic}?"}],
            backend="llamacpp",
            device="cuda",
            max_tokens=100
        )
        for topic in ["Python", "JavaScript", "Rust"]
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"\nResponse {i+1}:")
        print(response['choices'][0]['message']['content'])


if __name__ == "__main__":
    asyncio.run(async_example())
```

---

## Node.js Examples

### Basic Client

```javascript
const axios = require('axios');

class LLMClient {
  constructor(baseURL = 'http://localhost:8080') {
    this.baseURL = baseURL;
    this.client = axios.create({ baseURL });
  }

  async chat({
    model,
    messages,
    backend = 'llamacpp',
    device = 'cuda',
    stream = false,
    maxTokens = 512,
    temperature = 0.7,
    ...options
  }) {
    const payload = {
      model,
      backend,
      device,
      messages,
      stream,
      max_tokens: maxTokens,
      temperature,
      ...options
    };

    if (stream) {
      return this._streamResponse(payload);
    } else {
      const response = await this.client.post('/v1/chat/completions', payload);
      return response.data;
    }
  }

  async *_streamResponse(payload) {
    const response = await this.client.post('/v1/chat/completions', payload, {
      responseType: 'stream'
    });

    for await (const chunk of response.data) {
      const lines = chunk.toString().split('\n').filter(line => line.trim());

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);

          if (data === '[DONE]') {
            return;
          }

          try {
            const parsed = JSON.parse(data);
            yield parsed;
          } catch (e) {
            // Skip invalid JSON
          }
        }
      }
    }
  }

  async listModels() {
    const response = await this.client.get('/v1/models/loaded');
    return response.data;
  }

  async getStats() {
    const response = await this.client.get('/v1/models/stats');
    return response.data;
  }

  async downloadModel({ url, quantization, destination }) {
    const response = await this.client.post('/v1/models/download', {
      url,
      quantization,
      destination
    });
    return response.data;
  }

  async getDownloadStatus(jobId) {
    const response = await this.client.get(`/v1/models/download/${jobId}`);
    return response.data;
  }
}

module.exports = LLMClient;
```

### Example Usage

```javascript
const LLMClient = require('./llm-client');

async function basicExample() {
  const client = new LLMClient();

  try {
    const response = await client.chat({
      model: '/models/qwen2.5-coder-7b-instruct-gguf',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is Node.js?' }
      ],
      backend: 'llamacpp',
      device: 'cuda',
      maxTokens: 200
    });

    console.log('Response:', response.choices[0].message.content);
    console.log('Tokens used:', response.usage.total_tokens);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

basicExample();
```

### Streaming Example

```javascript
async function streamingExample() {
  const client = new LLMClient();

  console.log('Streaming response:\n');

  try {
    for await (const chunk of client.chat({
      model: '/models/qwen2.5-coder-7b-instruct-gguf',
      messages: [
        { role: 'user', content: 'Explain async/await in JavaScript' }
      ],
      backend: 'llamacpp',
      device: 'cuda',
      stream: true,
      maxTokens: 300
    })) {
      const delta = chunk.choices?.[0]?.delta;
      const content = delta?.content;

      if (content) {
        process.stdout.write(content);
      }
    }

    console.log('\n\nStreaming complete!');
  } catch (error) {
    console.error('Error:', error.message);
  }
}

streamingExample();
```

### Promise.all for Concurrent Requests

```javascript
async function concurrentRequests() {
  const client = new LLMClient();

  const questions = [
    'What is TypeScript?',
    'What is React?',
    'What is Express.js?'
  ];

  const promises = questions.map(question =>
    client.chat({
      model: '/models/qwen2.5-coder-7b-instruct-gguf',
      messages: [{ role: 'user', content: question }],
      backend: 'llamacpp',
      device: 'cuda',
      maxTokens: 100
    })
  );

  try {
    const responses = await Promise.all(promises);

    responses.forEach((response, index) => {
      console.log(`\nQuestion ${index + 1}: ${questions[index]}`);
      console.log('Answer:', response.choices[0].message.content);
      console.log('-'.repeat(50));
    });
  } catch (error) {
    console.error('Error:', error.message);
  }
}

concurrentRequests();
```

### Download with Progress

```javascript
async function downloadWithProgress(url, quantization) {
  const client = new LLMClient();

  // Start download
  const job = await client.downloadModel({ url, quantization });
  console.log(`Download started: ${job.job_id}`);
  console.log(`Destination: ${job.destination}`);

  // Poll for status
  while (true) {
    await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds

    const status = await client.getDownloadStatus(job.job_id);
    console.log(`Status: ${status.status} - ${status.message}`);

    if (status.status === 'completed') {
      console.log('\nDownload complete!');
      console.log(`Files downloaded: ${status.files_downloaded}`);
      console.log(`Duration: ${status.duration_seconds}s`);
      return status.destination;
    }

    if (status.status === 'failed') {
      throw new Error(`Download failed: ${status.error}`);
    }

    if (status.status === 'cancelled') {
      console.log('Download was cancelled');
      return null;
    }
  }
}

// Usage
downloadWithProgress(
  'https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',
  'Q4_K_M'
).then(modelPath => {
  console.log(`Model ready at: ${modelPath}`);
}).catch(error => {
  console.error('Error:', error.message);
});
```

---

## Common Patterns

### Retry Logic

```python
import time
from typing import Callable, Any

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Any:
    """
    Retry function with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for each retry
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {delay}s...")
                time.sleep(delay)
                delay *= backoff_factor
    
    raise last_exception


# Usage
client = LLMClient()

def make_request():
    return client.chat(
        model="/models/qwen2.5-coder-7b-instruct-gguf",
        messages=[{"role": "user", "content": "Hello"}],
        backend="llamacpp",
        device="cuda"
    )

response = retry_with_backoff(make_request, max_retries=3)
```

### Rate Limiting

```python
import time
from collections import deque

class RateLimiter:
    """Simple token bucket rate limiter"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove old requests outside time window
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # Check if at limit
        if len(self.requests) >= self.max_requests:
            wait_time = self.requests[0] + self.time_window - now
            if wait_time > 0:
                print(f"Rate limit reached, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                return self.wait_if_needed()
        
        # Record this request
        self.requests.append(now)


# Usage: Max 10 requests per minute
rate_limiter = RateLimiter(max_requests=10, time_window=60)

client = LLMClient()

for i in range(20):
    rate_limiter.wait_if_needed()
    
    response = client.chat(
        model="/models/qwen2.5-coder-7b-instruct-gguf",
        messages=[{"role": "user", "content": f"Request {i}"}],
        backend="llamacpp",
        device="cuda"
    )
    
    print(f"Request {i} completed")
```

### Conversation History Management

```python
class ConversationManager:
    """Manage conversation history with token limits"""
    
    def __init__(self, max_tokens: int = 2048):
        self.messages = []
        self.max_tokens = max_tokens
    
    def add_message(self, role: str, content: str):
        """Add message to history"""
        self.messages.append({"role": role, "content": content})
        self._truncate_if_needed()
    
    def _truncate_if_needed(self):
        """Remove old messages if exceeding token limit"""
        # Simple word-based estimation (4 chars per token)
        total_chars = sum(len(msg['content']) for msg in self.messages)
        estimated_tokens = total_chars // 4
        
        while estimated_tokens > self.max_tokens and len(self.messages) > 2:
            # Keep system message if present
            if self.messages[0]['role'] == 'system':
                self.messages.pop(1)
            else:
                self.messages.pop(0)
            
            total_chars = sum(len(msg['content']) for msg in self.messages)
            estimated_tokens = total_chars // 4
    
    def get_messages(self):
        """Get current message history"""
        return self.messages.copy()


# Usage
conv = ConversationManager(max_tokens=1000)
conv.add_message("system", "You are a helpful assistant.")

client = LLMClient()

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    conv.add_message("user", user_input)
    
    response = client.chat(
        model="/models/qwen2.5-coder-7b-instruct-gguf",
        messages=conv.get_messages(),
        backend="llamacpp",
        device="cuda"
    )
    
    assistant_message = response['choices'][0]['message']['content']
    conv.add_message("assistant", assistant_message)
    
    print(f"Assistant: {assistant_message}\n")
```

---

## Advanced Usage

### Multi-Backend Fallback

```python
def chat_with_fallback(client, messages, backends_priority):
    """Try multiple backends in order until one succeeds"""
    for backend_config in backends_priority:
        try:
            print(f"Trying {backend_config['backend']} on {backend_config['device']}...")
            
            response = client.chat(
                messages=messages,
                **backend_config
            )
            
            print(f"Success with {backend_config['backend']}!")
            return response
            
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    raise RuntimeError("All backends failed")


# Usage
client = LLMClient()

backends = [
    # Try GPU first
    {
        "model": "/models/qwen2.5-coder-7b-instruct-gguf",
        "backend": "llamacpp",
        "device": "cuda",
        "n_gpu_layers": 35
    },
    # Fallback to CPU
    {
        "model": "/models/qwen2.5-coder-7b-instruct-gguf",
        "backend": "llamacpp",
        "device": "cpu",
        "n_gpu_layers": 0
    },
    # Fallback to different model
    {
        "model": "/models/smaller-model",
        "backend": "transformers",
        "device": "cpu"
    }
]

response = chat_with_fallback(
    client,
    messages=[{"role": "user", "content": "Hello!"}],
    backends_priority=backends
)
```

### Batch Processing

```python
import concurrent.futures

def process_batch(client, items, model_config, max_workers=5):
    """Process multiple items in parallel"""
    def process_item(item):
        response = client.chat(
            messages=[{"role": "user", "content": item}],
            **model_config
        )
        return {
            "input": item,
            "output": response['choices'][0]['message']['content'],
            "tokens": response['usage']['total_tokens']
        }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_item, items))
    
    return results


# Usage
client = LLMClient()

items = [
    "Explain Python list comprehensions",
    "What are Python decorators?",
    "Explain Python generators",
    "What is a Python context manager?",
    "Explain Python metaclasses"
]

model_config = {
    "model": "/models/qwen2.5-coder-7b-instruct-gguf",
    "backend": "llamacpp",
    "device": "cuda",
    "max_tokens": 200
}

results = process_batch(client, items, model_config, max_workers=3)

for result in results:
    print(f"\nQ: {result['input']}")
    print(f"A: {result['output']}")
    print(f"Tokens: {result['tokens']}")
    print("-" * 80)
```

---

## Error Handling

### Comprehensive Error Handling

```python
import requests
from typing import Optional

def safe_chat(client, **kwargs) -> Optional[dict]:
    """Chat with comprehensive error handling"""
    try:
        response = client.chat(**kwargs)
        return response
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running?")
        return None
    
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Try increasing timeout or reducing max_tokens.")
        return None
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            print(f"Error: Invalid request - {e.response.json()}")
        elif e.response.status_code == 404:
            print(f"Error: Model not found. Check model path.")
        elif e.response.status_code == 500:
            print(f"Error: Server error - {e.response.json()}")
        else:
            print(f"Error: HTTP {e.response.status_code}")
        return None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# Usage
client = LLMClient()

response = safe_chat(
    client,
    model="/models/qwen2.5-coder-7b-instruct-gguf",
    messages=[{"role": "user", "content": "Hello"}],
    backend="llamacpp",
    device="cuda"
)

if response:
    print(response['choices'][0]['message']['content'])
else:
    print("Request failed, using fallback response")
```

---

## Production Deployment

### Docker Compose with Client

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  llm-server:
    image: yourusername/multi-llm-server:latest
    container_name: llm-server
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
    environment:
      - DEFAULT_TTL=600
      - CLEANUP_INTERVAL=60
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  client-app:
    build: ./client
    container_name: client-app
    depends_on:
      - llm-server
    environment:
      - LLM_SERVER_URL=http://llm-server:8080
    restart: unless-stopped
```

### Health Check Monitoring

```python
import requests
import time

def health_check(url: str, timeout: int = 5) -> bool:
    """Check if server is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def wait_for_server(url: str, max_wait: int = 60):
    """Wait for server to become ready"""
    print("Waiting for server to be ready...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if health_check(url):
            print("Server is ready!")
            return True
        
        time.sleep(2)
    
    print("Server failed to become ready")
    return False


# Usage in startup script
if __name__ == "__main__":
    server_url = "http://localhost:8080"
    
    if wait_for_server(server_url):
        client = LLMClient(server_url)
        # Start your application
    else:
        exit(1)
```

---

These examples should cover most common use cases. For more advanced scenarios, refer to the [API documentation](README.md#api-reference) and [architecture details](ARCHITECTURE.md).
