from flask import Flask, request, Response
from flask_cors import CORS  # Import flask_cors

app = Flask(__name__)
CORS(app)  # Initialize CORS with default settings

DNA_FILE_PATH = 'ref_dna.txt'

@app.route('/get_dna_segment', methods=['GET', 'OPTIONS'])
def get_dna_segment():
    try:
        start = int(request.args.get('start', 0))
        end = int(request.args.get('end', 0))

        if start < 0 or end <= start:
            return Response('Invalid range', status=400)

        length = end - start

        with open(DNA_FILE_PATH, 'r') as f:
            f.seek(start)
            dna_segment = f.read(length)

        return Response(dna_segment, mimetype='text/plain')
    except Exception as e:
        print(f"Error in /get_dna_segment: {e}")
        return Response('Server error', status=500)

@app.route('/test')
def test():
    return 'Test route is working!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


