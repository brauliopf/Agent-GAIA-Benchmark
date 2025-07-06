import os

def save_graph(graph, filename):
    # Save graph visualization to file
    graph_png = graph.get_graph().draw_mermaid_png()
    # Use absolute path based on current file location
    # current_dir gets the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(current_dir, filename)
    with open(graph_path, "wb") as f:
        f.write(graph_png)
