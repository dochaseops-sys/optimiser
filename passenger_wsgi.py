import sys
import os

# Add the application directory to the python path so imports resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from a2wsgi import ASGIMiddleware
from dbse.backend.main import app

# Wrap the ASGI FastAPI app as a WSGI application for Phusion Passenger
application = ASGIMiddleware(app)
