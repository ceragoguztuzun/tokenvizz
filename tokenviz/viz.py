import json
import csv

def read_edges_from_csv(file_path):
    """
    Read edges from a CSV file.
    
    :param file_path: Path to the CSV file containing edge data
    :return: List of tuples (source, target, weight) representing graph edges
    """
    edges = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            source, target, weight = int(row[0]), int(row[1]), float(row[2])
            edges.append((source, target, weight))
    return edges

def read_node_info_from_json(file_path):
    """
    Read node information from a JSON file.
    
    :param file_path: Path to the JSON file containing node information
    :return: Dictionary mapping node IDs to dictionaries containing 'string' and 'position'
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_graph_visualization(edges_file, node_info_file, ref_dna, output_file='graph_visualization.html'):
    """
    Generate an interactive graph visualization HTML file from edge data and node information files.
    
    :param edges_file: Path to the CSV file containing edge data
    :param node_info_file: Path to the JSON file containing node information
    :param ref_dna: Reference DNA sequence
    :param output_file: Name of the output HTML file
    """
    edges = read_edges_from_csv(edges_file)
    node_info = read_node_info_from_json(node_info_file)
    
    # Prepare nodes and edges data
    nodes = [{"id": i, "label": info['string'], "position": info['position']} for i, info in node_info.items()]
    edges = [{"from": s, "to": t, "label": f"{w:.3f}", "weight": w} for s, t, w in edges]
    
    # Calculate max weight for slider
    max_weight = max(edge['weight'] for edge in edges)
    
    # Create the HTML template using an f-string
    html_template = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DNA Tokenviz</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
        <style>
            body {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                padding: 0;
            }}
            #mynetwork {{
                width: 90vw;
                height: 70vh;
                border: 1px solid lightgray;
            }}
            .control-panel {{
                margin-top: 20px;
                width: 90vw;
                text-align: center;
            }}
            #node-info {{
                margin-top: 20px;
                font-weight: bold;
            }}
            #ref-dna {{
                font-family: monospace;
                font-size: 24px;
                margin-bottom: 20px;
                letter-spacing: 5px;
                white-space: nowrap;
                overflow-x: auto;
                padding: 10px;
                text-align: center;
            }}
            .highlight {{
                background-color: yellow;
            }}
            #search-bar {{
                margin-top: 10px;
            }}
            #change-background {{
                position: fixed;
                bottom: 20px;
                right: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>DNA Tokenviz</h1>
        <div id="ref-dna">{ref_dna}</div>
        <div class="control-panel">
            <label for="weight-slider">Edge Weight Threshold: </label>
            <input type="range" id="weight-slider" min="0" max="{max_weight}" step="0.001" value="0">
            <span id="slider-value">0</span>
        </div>
        <div id="search-bar">
            <input type="number" id="position-search" placeholder="Enter position">
            <button id="search-button">Search</button>
        </div>
        <div id="mynetwork"></div>
        <div id="node-info"></div>
        <button id="change-background">Galaxy Mode</button>

        <script>
            var nodes = new vis.DataSet({json.dumps(nodes)});
            var edges = new vis.DataSet({json.dumps(edges)});

            var container = document.getElementById('mynetwork');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            var options = {{
                nodes: {{
                    shape: 'box',
                    font: {{
                        size: 18
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
                }}
            }};
            var network = new vis.Network(container, data, options);

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
                var refDNA = document.getElementById('ref-dna');
                var dnaText = refDNA.textContent;
                var highlightedText = dnaText.substring(0, start) +
                    '<span class="highlight">' + dnaText.substring(start, end) + '</span>' +
                    dnaText.substring(end);
                refDNA.innerHTML = highlightedText;
                
                // Center the highlighted area
                var highlightedSpan = refDNA.querySelector('.highlight');
                if (highlightedSpan) {{
                    var containerWidth = refDNA.offsetWidth;
                    var spanWidth = highlightedSpan.offsetWidth;
                    var spanLeft = highlightedSpan.offsetLeft;
                    var textWidth = refDNA.scrollWidth;
                    
                    // Calculate the left position to center the highlighted text
                    var leftPosition = Math.max(0, spanLeft - (containerWidth / 2) + (spanWidth / 2));
                    
                    // Ensure the text doesn't scroll past its boundaries
                    leftPosition = Math.min(leftPosition, textWidth - containerWidth);
                    
                    // Apply the scroll
                    refDNA.scrollLeft = leftPosition;
                    
                    // Adjust vertical alignment if needed
                    var containerHeight = refDNA.offsetHeight;
                    var spanHeight = highlightedSpan.offsetHeight;
                    var spanTop = highlightedSpan.offsetTop;
                    refDNA.scrollTop = spanTop - (containerHeight / 2) + (spanHeight / 2);
                }}
            }}

            function clickNode(nodeId) {{
                var node = nodes.get(nodeId);
                document.getElementById('node-info').innerHTML = `Node: ${{node.label}}, Position: ${{node.position}}`;
                var [start, end] = node.position.split('-').map(Number);
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

            // Add click event to show node position and highlight DNA
            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    clickNode(params.nodes[0]);
                }}
            }});

            // Add button click event to toggle background
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

            // Add search functionality
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
        </script>
    </body>
    </html>
    '''

    # Write the HTML file
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"Graph visualization has been generated in '{output_file}'")

# Example usage
if __name__ == "__main__":
    #ref_dna = "AGCTTAGCTAGCTAGCTGACT"
    #read ref_dna from file
    with open("ref_dna.txt", "r") as f:
        ref_dna = f.read()
        
    edges_file = "attention_graph_nothreshold.csv"
    node_info_file = "node_info.json"
    generate_graph_visualization(edges_file, node_info_file, ref_dna)
