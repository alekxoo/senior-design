import time
import psutil
import logging
import argparse

# Setup logging for performance monitoring
logging.basicConfig(filename='performance_monitor.log', level=logging.INFO)

def log_performance(cpu_usage, memory_usage):
    logging.info(f"CPU: {cpu_usage}%, Memory: {memory_usage}%")
    print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

def monitor_system(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        # Log CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        log_performance(cpu_usage, memory_usage)
        time.sleep(1)  # Log every 1 second

def main():
    parser = argparse.ArgumentParser(description="Monitor system performance while running an application")
    parser.add_argument('--duration', type=int, default=60, help="Duration to monitor in seconds")
    args = parser.parse_args()

    print(f"Monitoring system performance for {args.duration} seconds...")
    monitor_system(args.duration)

if __name__ == "__main__":
    main()
