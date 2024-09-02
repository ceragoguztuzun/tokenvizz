import json

def generate_graph_visualization(edges, node_info, output_file='graph_visualization.html'):
    """
    Generate an interactive graph visualization HTML file from edge data and node information.
    
    :param edges: List of tuples (source, target, weight) representing graph edges
    :param node_info: Dictionary mapping node IDs to dictionaries containing 'string' and 'position'
    :param output_file: Name of the output HTML file
    """
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
        <title>Interactive Graph Visualization Tool</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
        <style>
            #mynetwork {{
                width: 600px;
                height: 400px;
                border: 1px solid lightgray;
            }}
            .control-panel {{
                margin-top: 20px;
            }}
            #node-info {{
                margin-top: 20px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>Interactive Graph Visualization Tool</h1>
        <div class="control-panel">
            <label for="weight-slider">Edge Weight Threshold: </label>
            <input type="range" id="weight-slider" min="0" max="{max_weight}" step="0.001" value="0">
            <span id="slider-value">0</span>
        </div>
        <div id="mynetwork"></div>
        <div id="node-info"></div>

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

            // Add click event to show node position
            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    document.getElementById('node-info').innerHTML = `Node: ${{node.label}}, Position: ${{node.position}}`;
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
    # Example data
    edges = [
        (0, 1, 0.30486584),
        (0, 2, 0.18741572),
        (0, 3, 0.11908629),
        (0, 4, 0.09191545),
        (0, 5, 0.1012267),
        (1, 2, 0.21566181),
        (1, 3, 0.14147946),
        (1, 4, 0.10689311),
        (1, 5, 0.10828577),
        (2, 3, 0.28788823),
        (2, 4, 0.14343601),
        (2, 5, 0.12260079),
        (3, 4, 0.16437443),
        (3, 5, 0.118000805),
        (4, 5, 0.1553517)
    ]
    
    node_info = {
        0: {"string": "A", "position": "0-1"},
        1: {"string": "GCTTA", "position": "1-6"},
        2: {"string": "GCTA", "position": "2-6"},
        3: {"string": "GCTA", "position": "3-7"},
        4: {"string": "GCTGA", "position": "4-9"},
        5: {"string": "CT", "position": "5-7"}
    }
    
    generate_graph_visualization(edges, node_info)
