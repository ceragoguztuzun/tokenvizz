from flask import Flask, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Path to your DNA sequence file
DNA_FILE_PATH = 'ref_dna.txt'

@app.route('/get_dna_segment')
def get_dna_segment():
    start = int(request.args.get('start', 0))
    end = int(request.args.get('end', 0))

    if start < 0 or end <= start:
        return Response('Invalid range', status=400)

    length = end - start

    try:
        with open(DNA_FILE_PATH, 'r') as f:
            f.seek(start)
            dna_segment = f.read(length)
    except Exception as e:
        print(f"Error reading DNA segment: {e}")
        return Response('Error reading DNA segment', status=500)

    return Response(dna_segment, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
