from flask import Flask, request, Response
from flask_cors import CORS
import pysam
import logging
import os

app = Flask(__name__)
CORS(app)  # Initialize CORS with default settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Path to the FASTA file
FASTA_FILE_PATH = '/usr/homes/cxo147/ceRAG_viz/data/hg38.fa'

# Initialize the pysam FastaFile once to avoid reopening it on every request
try:
    fasta = pysam.FastaFile(FASTA_FILE_PATH)
    # If multiple references exist, specify the desired reference name here
    REFERENCE_NAME = fasta.references[0]  # Change as needed
    logging.info(f"Loaded FASTA file: {FASTA_FILE_PATH}")
    #logging.info(f"Reference sequences available: {fasta.references}")
    logging.info(f"Using reference: {REFERENCE_NAME}")
except Exception as e:
    logging.error(f"Error loading FASTA file: {e}")
    fasta = None
    REFERENCE_NAME = None

@app.route('/get_dna_segment', methods=['GET', 'OPTIONS'])
def get_dna_segment():
    if fasta is None or REFERENCE_NAME is None:
        return Response('FASTA file not loaded properly.', status=500)
    
    try:
        # Get 'start' and 'end' parameters from query string
        start = request.args.get('start')
        end = request.args.get('end')
        print('start and end', start, end)

        if start is None or end is None:
            return Response('Missing start or end parameters.', status=400)
        
        # Convert to integers
        start = int(start)
        end = int(end)
        print('start and end', start, end)
        print('->', end-start)

        # Validate the range
        if start < 0 or end <= start:
            return Response('Invalid range: start must be >= 0 and end must be > start.', status=400)
        
        # Fetch the DNA segment using pysam
        # pysam uses 0-based, half-open intervals [start, end)
        dna_segment = fasta.fetch(REFERENCE_NAME, start, end)
        print(dna_segment)

        # Return the DNA segment as plain text
        return Response(dna_segment, mimetype='text/plain')
    
    except ValueError:
        return Response('Start and end parameters must be integers.', status=400)
    except Exception as e:
        logging.error(f"Error in /get_dna_segment: {e}")
        logging.error(traceback.format_exc())
        return Response('Server error', status=500)

@app.route('/test')
def test():
    return 'Test route is working!'

if __name__ == '__main__':
    if fasta is None:
        logging.error("FASTA file could not be loaded. Exiting.")
        exit(1)
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
