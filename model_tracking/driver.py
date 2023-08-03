from models_scanner import perform_tracking
import datetime
import time

while True:
  perform_tracking(str(datetime.datetime.now()))
  time.sleep(30)
