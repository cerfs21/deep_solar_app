# Bridging application to server using WSGI
import sys
sys.path.insert(0,"/var/www/deep_solar_app/")
from deep_solar_app import server as application
