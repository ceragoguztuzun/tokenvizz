import json
import csv
import scipy.sparse as sp
import os
import argparse

def read_edges_from_csv(file_path):
    edges = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            source, target, weight = int(row[0]), int(row[1]), float(row[2])
            edges.append((source, target, weight))
    return edges

def read_edges_from_npz(file_path, unique=True):
    adjacency_matrix = sp.load_npz(file_path)
    
    if not sp.isspmatrix_coo(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocoo()
    
    if unique:
        mask = adjacency_matrix.row <= adjacency_matrix.col
        sources = adjacency_matrix.row[mask]
        targets = adjacency_matrix.col[mask]
        weights = adjacency_matrix.data[mask]
    else:
        sources = adjacency_matrix.row
        targets = adjacency_matrix.col
        weights = adjacency_matrix.data
    
    edges = list(zip(sources.tolist(), targets.tolist(), weights.tolist()))
    
    return edges

def read_node_info_from_json(file_path):
    with open(file_path, 'r') as f:
        node_info = json.load(f)
    return node_info

def extract_reference_from_edges_file(edges_file):
    # Extract the filename from the file path
    filename = os.path.basename(edges_file)
    
    # Assuming the format is always {ref-name}_attention_graph.npz
    reference = filename.split('_')[0]
    
    return reference

def generate_graph_visualization(edges_file, node_info_file, output_file='graph_vizzy.html', default_weight=0.0):
    edges = read_edges_from_npz(edges_file)
    node_info = read_node_info_from_json(node_info_file)

    # Extract the reference dynamically from the edges file
    reference = extract_reference_from_edges_file(edges_file)

    # Example model name, can be passed as a parameter if needed
    model_name = 'jaandoui/DNABERT2-AttentionExtracted'

    # Include the 'color' and 'weighted_degree' fields for each node
    nodes = []
    for i, info in node_info.items():
        node = {
            "id": int(i),
            "label": info['string'],
            "position": info['position'],
            "color": info.get('color', '#97C2FC'),  # Use the color from node_info.json
            "title": f"Weighted Degree: {info.get('weighted_degree', 0):.3f}"
        }
        nodes.append(node)

    edges = [{"from": int(s), "to": int(t), "label": f"{w:.3f}", "weight": w} for s, t, w in edges]
    # Use default_weight for both minimum and default value of the slider
    min_weight = default_weight
    max_weight = max(edge['weight'] for edge in edges) if edges else 0

    # Insert model_name and reference into the HTML template
        # Insert model_name and reference into the HTML template
        # Insert model_name and reference into the HTML template
        # Insert model_name and reference into the HTML template
        # Insert model_name and reference into the HTML template
        # Insert model_name and reference into the HTML template
        # Insert model_name and reference into the HTML template
    html_template = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>DNA Tokenviz</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
        <style>
            body {{
                display: flex;
                flex-direction: column;
                margin: 0;
                padding: 0;
                height: 100vh;
                font-family: Arial, sans-serif;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                background-color: #f0f0f0;
            }}
            .header h1 {{
                margin: 0;
            }}
            .controls {{
                display: flex;
                align-items: center;
            }}
            .control-item {{
                margin-left: 20px;
            }}
            .control-item label {{
                font-weight: bold; /* Make labels bold */
            }}
            .control-item img {{
                vertical-align: middle; /* Align image with text */
                width: 24px; /* Set image width */
                height: 16px; /* Set image height */
                margin-right: 5px; /* Space between image and text */
            }}
            .container {{
                display: flex;
                width: 100%;
                height: calc(100vh - 60px);
            }}
            .left-panel {{
                flex: 2;
                display: flex;
                flex-direction: column;
                padding: 10px;
                border-right: 1px solid #ccc;
            }}
            .right-panel {{
                flex: 1;
                display: flex;
                flex-direction: column;
                padding: 10px;
            }}
            #ref-dna {{
                font-family: monospace;
                font-size: 18px;
                letter-spacing: 2px;
                white-space: pre-wrap;
                overflow-y: auto;
                padding: 10px;
                flex-grow: 1;
                border: 1px solid #ccc;
                box-sizing: border-box;
            }}
            .highlight {{
                background-color: yellow;
            }}
            .highlight {{
                transition: all 0.3s ease-in-out;
            }}
            .highlight:hover {{
                background-color: rgba(255, 255, 0, 0.4); /* More transparent on hover */
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.3); /* Add a subtle shadow to make it pop */
                transform: scale(1.05); /* Slightly increase the size */
                border-color: rgba(0, 0, 0, 0.8); /* Make the border a bit darker */
            }}
            #mynetwork {{
                width: 100%;
                height: 80vh;
                border: 1px solid lightgray;
                background-color: white; /* Default background */
                background-size: cover;
            }}
            #node-info {{
                margin-top: 20px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>DNA Tokenviz</h1>
            <div class="controls">
                <div class="control-item">
                    <label for="ref-display">Reference: </label>
                    <span id="ref-display">{reference}</span>
                </div>
                <div class="control-item">
                    <label for="model-display">
                        <img src="hgface.png" alt="Model icon">Model: 
                    </label>
                    <span id="model-display">{model_name}</span>
                </div>
                <div class="control-item">
                    <label for="weight-slider">Edge Weight: </label>
                    <input type="range" id="weight-slider" min="{min_weight}" max="{max_weight}" step="0.001" value="{default_weight}">
                    <span id="slider-value">{default_weight:.3f}</span>
                </div>
                <div class="control-item">
                    <input type="number" id="position-search" placeholder="Enter position">
                    <button id="search-button">Search</button>
                </div>
            </div>
        </div>
        <div class="container">
            <div class="left-panel">
                <div id="mynetwork"></div>
                <div id="node-info"></div>
            </div>
            <div class="right-panel">
                <div id="ref-dna"></div>
            </div>
        </div>

        <script>
            // Define isColorDark function
            function isColorDark(color) {{
                var r, g, b;
                if (color.startsWith('#')) {{
                    // Convert hex to RGB
                    var hex = color.substring(1);
                    if (hex.length === 3) {{
                        r = parseInt(hex[0] + hex[0], 16);
                        g = parseInt(hex[1] + hex[1], 16);
                        b = parseInt(hex[2] + hex[2], 16);
                    }} else {{
                        r = parseInt(hex.substring(0, 2), 16);
                        g = parseInt(hex.substring(2, 4), 16);
                        b = parseInt(hex.substring(4, 6), 16);
                    }}
                }} else {{
                    return false; // Unsupported color format
                }}
                // Calculate brightness
                var brightness = (r * 299 + g * 587 + b * 114) / 1000;
                return brightness < 128; // Consider it dark if brightness is below 128
            }}

            // Function to convert hex color to rgba with transparency
            function hexToRgba(hex, alpha) {{
                var r = 0, g = 0, b = 0;
                hex = hex.replace('#', '');

                if (hex.length === 3) {{
                    r = parseInt(hex[0] + hex[0], 16);
                    g = parseInt(hex[1] + hex[1], 16);
                    b = parseInt(hex[2] + hex[2], 16);
                }} else if (hex.length === 6) {{
                    r = parseInt(hex.substring(0, 2), 16);
                    g = parseInt(hex.substring(2, 4), 16);
                    b = parseInt(hex.substring(4, 6), 16);
                }}

                return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
            }}

            var rawNodes = {json.dumps(nodes)};

            // Apply font color adjustment for dark-colored nodes
            rawNodes.forEach(function (node) {{
                if (isColorDark(node.color)) {{
                    node.font = {{ color: '#ffffff' }}; // Set font color to white for dark nodes
                }}
            }});

            var nodes = new vis.DataSet(rawNodes);
            var edges = new vis.DataSet({json.dumps(edges)});

            var container = document.getElementById('mynetwork');

            var options = {{
                nodes: {{
                    shape: 'box',
                    font: {{
                        size: 18
                    }},
                    color: {{
                        border: '#000000',
                        highlight: {{
                            border: '#2B7CE9',
                            background: '#D2E5FF'
                        }},
                        hover: {{
                            border: '#2B7CE9',
                            background: '#D2E5FF'
                        }}
                    }}
                }},
                edges: {{
                    font: {{
                        size: 12
                    }}
                }},
                physics: {{
                    stabilization: false,
                    barnesHut: {{
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 150,
                        springConstant: 0.04,
                        damping: 0.09,
                        avoidOverlap: 0.1
                    }}
                }},
                interaction: {{
                    tooltipDelay: 200,
                    hover: true
                }}
            }};

            var network = new vis.Network(container, {{nodes: nodes, edges: edges}}, options);

            // Filter edges based on edge weight
            function filterEdges(threshold) {{
                var filteredEdges = edges.get().filter(function(edge) {{
                    return edge.weight >= threshold;
                }});
                network.setData({{
                    nodes: nodes,
                    edges: filteredEdges
                }});
            }}

            // Update the displayed weight when the slider is changed
            var slider = document.getElementById('weight-slider');
            var sliderValue = document.getElementById('slider-value');
            slider.oninput = function() {{
                var value = parseFloat(this.value).toFixed(3);
                sliderValue.innerHTML = value;
                filterEdges(parseFloat(value));
            }}

            function clickNode(nodeId) {{
                var node = nodes.get(nodeId);
                document.getElementById('node-info').innerHTML = `Node: ${{node.label}}, Position: ${{node.position}}`;

                var position = node.position;
                if (!position || !position.includes('-')) {{
                    console.error('Invalid node position:', position);
                    return;
                }}

                var [startStr, endStr] = position.split('-').map(s => s.trim());
                var start = Number(startStr);
                var end = Number(endStr);
                var connectedNodes = network.getConnectedNodes(nodeId);

                if (isNaN(start) || isNaN(end)) {{
                    console.error('Invalid start or end positions:', start, end);
                    return;
                }}

                highlightDNA(start, end, node.color, connectedNodes);
                network.focus(nodeId, {{
                    scale: 1.5,
                    animation: {{
                        duration: 1000,
                        easingFunction: 'easeInOutQuad'
                    }}
                }});
                network.selectNodes([nodeId]);
            }}

            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    clickNode(params.nodes[0]);
                }}
            }});

            function highlightDNA(start, end, color, connectedNodes) {{
                var segmentLength = 200;
                var displayStart = Math.max(0, start - segmentLength);
                var displayEnd = end + segmentLength;

                fetch(`http://127.0.0.1:5000/get_dna_segment?start=${{displayStart}}&end=${{displayEnd}}`)
                    .then(response => response.text())
                    .then(dnaSegment => {{
                        var formattedDNA = dnaSegment.replace(/(.{{50}})/g, `$1\\n`);

                        var highlights = [];
                        var highlightStart = start - displayStart + 2;
                        var highlightEnd = end - displayStart + 3;
                        highlights.push({{
                            start: highlightStart,
                            end: highlightEnd,
                            style: `background-color: ${{hexToRgba(color, 0.3)}}; padding: 2px;` /* Transparent version of node color */
                        }});

                        connectedNodes.forEach(function(connectedNodeId) {{
                            var connectedNode = nodes.get(connectedNodeId);
                            var [connectedStartStr, connectedEndStr] = connectedNode.position.split('-').map(s => s.trim());
                            var connectedStart = Number(connectedStartStr);
                            var connectedEnd = Number(connectedEndStr);

                            if (connectedStart >= displayStart && connectedEnd <= displayEnd) {{
                                var connectedHighlightStart = connectedStart - displayStart + 2;
                                var connectedHighlightEnd = connectedEnd - displayStart + 2;
                                highlights.push({{
                                    start: connectedHighlightStart,
                                    end: connectedHighlightEnd,
                                    style: `background-color: ${{hexToRgba(connectedNode.color, 0.3)}}; padding: 2px;` /* Transparent version of connected node color */
                                }});
                            }}
                        }});

                        highlights.sort((a, b) => a.start - b.start);

                        var highlightedText = "";
                        var currentIndex = 0;
                        highlights.forEach(function(highlight) {{
                            highlightedText += formattedDNA.substring(currentIndex, highlight.start);
                            highlightedText += `<span class="highlight" style="${{highlight.style}}">`;
                            highlightedText += formattedDNA.substring(highlight.start, highlight.end + 1);
                            highlightedText += `</span>`;
                            currentIndex = highlight.end + 1;
                        }});

                        highlightedText += formattedDNA.substring(currentIndex);

                        document.getElementById('ref-dna').innerHTML = highlightedText;
                        var highlightElement = document.getElementById('ref-dna').querySelector('.highlight');
                        if (highlightElement) {{
                            highlightElement.scrollIntoView({{
                                behavior: 'smooth',
                                block: 'center',
                                inline: 'center'
                            }});
                        }}
                    }})
                    .catch(error => {{
                        document.getElementById('ref-dna').innerHTML = 'Error loading DNA segment.';
                    }});
            }}

            // Position search functionality
            document.getElementById('search-button').addEventListener('click', function() {{
                var position = parseInt(document.getElementById('position-search').value);
                var foundNode = nodes.get().find(node => {{
                    var [start, end] = node.position.split('-').map(Number);
                    return position >= start && position <= end;
                }});
                if (foundNode) {{
                    clickNode(foundNode.id);
                }} else {{
                    alert('No node found at this position in the visualized graph.');
                }}
            }});

        </script>
    </body>
    </html>
    '''

    with open(output_file, 'w') as f:
        f.write(html_template)

    print(f"Graph visualization has been generated in '{output_file}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DNA graph visualization")
    parser.add_argument('--model_name', type=str, default='jaandoui/DNABERT2-AttentionExtracted',
                        help='The name of the model to use (from huggingface)')
    parser.add_argument('--edges_file', type=str, required=True,
                        help='Path to the edges file (adjacency matrix in npz format)')
    parser.add_argument('--node_info_file', type=str, required=True,
                        help='Path to the node info file (node information in json format)')
    parser.add_argument('--output_file', type=str, default='graph_vizzy.html',
                        help='The output HTML file for the graph visualization')
    parser.add_argument('--default_weight', type=float, default=0.01,
                        help='The edge weight for the graph visualization (edge weight threshold you used when generating the graph)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Pass the arguments to the generate_graph_visualization function
    generate_graph_visualization(
        edges_file=args.edges_file,
        node_info_file=args.node_info_file,
        output_file=args.output_file,
        default_weight=args.default_weight
    )
