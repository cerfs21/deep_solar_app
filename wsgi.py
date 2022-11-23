# wsgi v1.1:
#   update file path

# Bridging application to server using WSGI
import sys
sys.path.insert(0,"/var/www/deep-solar/")
from deep_solar_app import server as application
