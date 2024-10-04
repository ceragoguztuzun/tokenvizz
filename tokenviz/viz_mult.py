import json
import csv
import scipy.sparse as sp

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

def generate_graph_visualization(edges_file, node_info_file, output_file='graph_vizzy.html'):
    edges = read_edges_from_npz(edges_file)
    node_info = read_node_info_from_json(node_info_file)

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

    max_weight = max(edge['weight'] for edge in edges) if edges else 0

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
            #mynetwork {{
                width: 100%;
                height: 80vh;
                border: 1px solid lightgray;
            }}
            #node-info {{
                margin-top: 20px;
                font-weight: bold;
            }}
            #change-background {{
                position: fixed;
                bottom: 20px;
                right: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>DNA Tokenviz</h1>
            <div class="controls">
                <div class="control-item">
                    <label for="tokenizer-select">Tokenizer: </label>
                    <select id="tokenizer-select">
                        <option value="tiktoken">TikToken</option>
                        <option value="bpe">BPE</option>
                        <option value="wordpiece">WordPiece</option>
                    </select>
                </div>
                <div class="control-item">
                    <label for="model-select">Model: </label>
                    <select id="model-select">
                        <option value="DNABERT2">DNABERT2</option>
                        <option value="NT">NT</option>
                        <option value="GeneBERT">GeneBERT</option>
                    </select>
                </div>
                <div class="control-item">
                    <label for="weight-slider">Edge Weight: </label>
                    <input type="range" id="weight-slider" min="0" max="{max_weight}" step="0.001" value="0.01">
                    <span id="slider-value">0.01</span>
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
        <button id="change-background">Galaxy Mode</button>

        <script>
            var nodes = new vis.DataSet({json.dumps(nodes)});
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

            function filterEdges(threshold) {{
                var filteredEdges = edges.get().filter(function(edge) {{
                    return edge.weight >= threshold;
                }});
                network.setData({{
                    nodes: nodes,
                    edges: filteredEdges
                }});
            }}

            var slider = document.getElementById('weight-slider');
            var sliderValue = document.getElementById('slider-value');
            slider.max = {max_weight};
            slider.step = {max_weight} / 1000;
            slider.oninput = function() {{
                var value = parseFloat(this.value).toFixed(3);
                sliderValue.innerHTML = value;
                filterEdges(parseFloat(value));
            }}

            function highlightDNA(start, end) {{
                console.log(`highlightDNA-> Pos DNA from ${{start}} to ${{end}}`);
                var segmentLength = 200;
                var displayStart = Math.max(0, start - segmentLength);
                var displayEnd = end + segmentLength;
                console.log(`highlightDNA-> displaying from ${{displayStart}} to ${{displayEnd}}`);

                fetch(`http://127.0.0.1:5000/get_dna_segment?start=${{displayStart}}&end=${{displayEnd}}`)
                    .then(response => {{
                        if (!response.ok) {{
                            throw new Error(`Server error: ${{response.status}}`);
                        }}
                        return response.text();
                    }})
                    .then(dnaSegment => {{
                        var highlightStart = start - displayStart + 2; // Adjust to highlight correctly
                        var highlightEnd = end - displayStart + 3;     // Adjust to include the full segment
                        console.log(`highlightDNA-> Highlighted DNA from ${{highlightStart}} to ${{highlightEnd}}`);

                        // Insert line breaks every 50 characters
                        var formattedDNA = dnaSegment.replace(/(.{{50}})/g, `$1\n`);

                        var highlightedText = `${{formattedDNA.substring(0, highlightStart)}}<span class="highlight">${{formattedDNA.substring(highlightStart, highlightEnd + 1)}}</span>${{formattedDNA.substring(highlightEnd + 1)}}`;

                        var refDNA = document.getElementById('ref-dna');
                        refDNA.innerHTML = highlightedText;

                        // Scroll to the highlighted part
                        var highlightElement = refDNA.querySelector('.highlight');
                        if (highlightElement) {{
                            highlightElement.scrollIntoView({{
                                behavior: 'smooth',
                                block: 'center',
                                inline: 'center'
                            }});
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error fetching DNA segment:', error);
                        var refDNA = document.getElementById('ref-dna');
                        refDNA.innerHTML = 'Error loading DNA segment.';
                    }});
            }}

            
            function clickNode(nodeId) {{
                var node = nodes.get(nodeId);
                console.log('clickNode-> Node clicked:', node);
                document.getElementById('node-info').innerHTML = `Node: ${{node.label}}, Position: ${{node.position}}`;

                var position = node.position;
                console.log('clickNode-> Node position:', position);

                if (!position || !position.includes('-')) {{
                    console.error('Invalid node position:', position);
                    return;
                }}

                var [startStr, endStr] = position.split('-').map(s => s.trim());
                var start = Number(startStr);
                var end = Number(endStr);

                if (isNaN(start) || isNaN(end)) {{
                    console.error('Invalid start or end positions:', start, end);
                    return;
                }}

                console.log('clickNode-> Parsed start:', start, 'Parsed end:', end);

                highlightDNA(start, end);
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

            var galaxyModeOn = false;
            document.getElementById('change-background').addEventListener('click', function() {{
                if (galaxyModeOn) {{
                    container.style.backgroundImage = "none";
                    galaxyModeOn = false;
                }} else {{
                    container.style.backgroundImage = "url('hxh.gif')";
                    container.style.backgroundSize = "100% 100%";
                    container.style.backgroundPosition = "center";
                    container.style.backgroundRepeat = "no-repeat";
                    galaxyModeOn = true;
                }}
            }});

            document.getElementById('search-button').addEventListener('click', function() {{
                var position = parseInt(document.getElementById('position-search').value);
                var foundNode = nodes.get().find(node => {{
                    var [start, end] = node.position.split('-').map(Number);
                    return position >= start && position < end;
                }});
                if (foundNode) {{
                    clickNode(foundNode.id);
                }} else {{
                    alert('No node found at this position');
                }}
            }});

            // Add event listeners for new dropdowns
            document.getElementById('tokenizer-select').addEventListener('change', function() {{
                console.log('Tokenizer changed to:', this.value);
                // Add logic to handle tokenizer change
            }});

            document.getElementById('model-select').addEventListener('change', function() {{
                console.log('Model changed to:', this.value);
                // Add logic to handle model change
            }});
        </script>
    </body>
    </html>
    '''

    with open(output_file, 'w') as f:
        f.write(html_template)

    print(f"Graph visualization has been generated in '{output_file}'")

# Example usage
if __name__ == "__main__":
    edges_file = "/usr/homes/cxo147/ceRAG_viz/tokenviz/outputs/adjacency_matrices/chr1_attention_graph.npz"
    node_info_file = "/usr/homes/cxo147/ceRAG_viz/tokenviz/outputs/node_info/chr1_node_info.json"
    generate_graph_visualization(edges_file, node_info_file, output_file='graph_vizzy.html')