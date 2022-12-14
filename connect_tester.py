# connect_tester v1.1:
#   create application path as a argument of create_session

'''
Unit test to check connection to SQLite database
'''

from datetime import datetime
from connect import create_session, get_last_connection_date, register_new_connection

app_path = '/var/www/deep-solar/'

def test_last_connection_date():
    start_date = datetime.now()
    name = "Test"

    session = create_session(app_path)
    register_new_connection(session, name)
    last_date = get_last_connection_date(session, name)

    # Check connection occurred by verifying that last_date has been assigned a value
    assert last_date is not None
    # Check connection date (last_date) is consistent with current date (start_date)
    assert last_date >= start_date

if __name__ == "__main__":
    test_last_connection_date()
    # Returns tests OK if all assert conditions are met
    print("All SQLite database tests ok.")
