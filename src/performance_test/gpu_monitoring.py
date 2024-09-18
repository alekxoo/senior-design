# Add GPU monitoring with jtop for Jetson Nano
from jtop import jtop

def monitor_system(duration):
    with jtop() as jetson:
        start_time = time.time()
        while time.time() - start_time < duration:
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent

            # GPU usage (specific to Jetson Nano)
            gpu_usage = jetson.gpu['val']

            logging.info(f"CPU: {cpu_usage}%, Memory: {memory_usage}%, GPU: {gpu_usage}%")
            print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%, GPU Usage: {gpu_usage}%")

            time.sleep(1)
