"""LangGraph graph — StateGraph definition and compilation."""

# TODO Day 3: Build StateGraph with nodes and conditional edges
# Graph flow:
#   [START] → router_node
#     ├── "qa"     → rag_node → rerank_node → synthesizer → [END]
#     ├── "search" → tool_node (search_anime) → synthesizer → [END]
#     └── "detail" → tool_node (get_details) → synthesizer → [END]
