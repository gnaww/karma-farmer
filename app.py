from app import app, socketio
from app.db import populate_db

if __name__ == "__main__":
    print("Populating database")
    populate_db()
    
    print("Flask app running at http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
