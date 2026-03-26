# NL2KG

Grammar-Constrained LLM Generation for Bidirectional Natural-Language Interaction with Robot Knowledge Graphs.

NL2KG is a ROS 2 system that provides a bidirectional pipeline between natural language and a distributed knowledge graph. It uses a locally-deployed LLM (Qwen3-4B via [llama_ros](https://github.com/mgonzs13/llama_ros)) with GBNF grammar-constrained decoding, ensuring every output is a syntactically valid JSON object encoding KG operations. An optional RAG pipeline with embedding retrieval and reranking improves context quality.

## Dependencies

- ROS 2 (Humble/Jazzy/Rolling)
- [llama_ros](https://github.com/mgonzs13/llama_ros) — Local LLM inference via llama.cpp
- [knowledge_graph](https://github.com/mgonzs13/knowledge_graph) — Distributed knowledge graph for ROS 2
- Python 3.10+

## Packages

| Package         | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| `nl2kg`         | Core ROS 2 node, CLI, models, grammar, metrics, and evaluation           |
| `nl2kg_msgs`    | ROS 2 action and message definitions (`NL2KG.action`, `KGOperation.msg`) |
| `nl2kg_bringup` | Launch files and parameter configs (LLM, embedding, reranker)            |

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select nl2kg_msgs nl2kg nl2kg_bringup
source install/setup.bash
```

## Package Structure

```
nl2kg/
├── grammars/
│   └── nl2kg.gbnf              # GBNF grammar for constrained JSON generation
├── nl2kg/
│   ├── __init__.py
│   ├── kg_context.py           # KG → text serializer for LLM context
│   ├── metrics.py              # Evaluation metrics (F1, accuracy, exact match)
│   ├── models.py               # Pydantic models (KGResponse, KGOperation)
│   ├── nl2kg_cli.py            # Interactive CLI client (action client)
│   ├── nl2kg_node.py           # Main ROS 2 node (action server)
│   └── utils.py                # Value casting utilities
├── package.xml
├── setup.py
└── setup.cfg

nl2kg_msgs/
├── action/
│   └── NL2KG.action            # ROS 2 action definition
├── msg/
│   └── KGOperation.msg         # KG operation message
├── CMakeLists.txt
└── package.xml

nl2kg_bringup/
├── launch/
│   └── nl2kg.launch.py         # Launch LLM + embedding + reranker + NL2KG node
├── params/
│   ├── nl2kg.yaml              # NL2KG node parameters
│   ├── llm.yaml                # Chat LLM config
│   ├── bge-base-en-v1.5.yaml   # Embedding model config
│   └── bge-reranker-v2-m3.yaml # Reranker model config
├── package.xml
└── setup.py
```

## ROS 2 Action Interface

NL2KG uses a ROS 2 action (`NL2KG.action`) instead of topics:

```
# Request
string input_text
---
# Result
bool success
string intent
nl2kg_msgs/KGOperation[] operations
string response
---
# Feedback
string status
```

The action server runs at `~/nl2kg` (default: `/nl2kg_node/nl2kg`).

## RAG Pipeline

When `enable_rag` is set to `true`, the node uses a retrieval-augmented generation pipeline:

1. **Indexing** — The current KG state is indexed into a Chroma vector store using BGE embeddings (`LlamaROSEmbeddings`)
2. **Retrieval** — The top-k documents are retrieved based on the user query
3. **Reranking** — A BGE reranker (`LlamaROSReranker`) re-scores and filters the retrieved documents using `ContextualCompressionRetriever`
4. **Generation** — The reranked context is injected into the system prompt for the LLM

## Running

### 1. Launch the NL2KG system

This starts the chat LLM, embedding model, reranker model, and the NL2KG node:

```bash
ros2 launch nl2kg_bringup nl2kg.launch.py
```

To use a different chat model:

```bash
ros2 launch nl2kg_bringup nl2kg.launch.py model_params:=/path/to/your/model.yaml
```

### 2. Interactive CLI

In a separate terminal, run the CLI to interact with the system:

```bash
ros2 run nl2kg nl2kg_cli
```

Type natural language commands like:

- `"Add a robot called robot1"` → creates a node
- `"robot1 is in the kitchen"` → creates an edge
- `"Set robot1 battery to 80"` → sets a property
- `"Where is robot1?"` → queries the graph
- `"Remove robot1 from the graph"` → deletes a node

### 3. ROS 2 Action Client

You can also send goals from the command line:

```bash
ros2 action send_goal /nl2kg_node/nl2kg nl2kg_msgs/action/NL2KG "{input_text: 'Add a robot called tiago'}" --feedback
```

## NL2KG Node Parameters

Configured in `nl2kg_bringup/params/nl2kg.yaml`:

| Parameter     | Default | Description                                    |
| ------------- | ------- | ---------------------------------------------- |
| `temperature` | `0.2`   | LLM sampling temperature                       |
| `use_gbnf`    | `true`  | Enable GBNF grammar-constrained decoding       |
| `enable_rag`  | `false` | Enable RAG with Chroma + embeddings + reranker |
