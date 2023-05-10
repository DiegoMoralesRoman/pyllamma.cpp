import time
import sys
import threading

class ProgressBar:
    """
    A progress bar class for iterating over an iterable
    """
    def __init__(self, iterable, prefix='', suffix='', fill='▁▂▃▄▅▆▇█', width=30):
        """
        Initializes the progress bar instance
        @param iterable: The iterable to be iterated over
        @param prefix: A string to be printed before the progress bar
        @param suffix: A string to be printed after the progress bar
        @param fill: A string to be used to fill the progress bar
        @param width: The width of the progress bar in characters
        """
        self.iterable = iterable
        self.total = len(iterable)
        self.prefix = prefix
        self.suffix = suffix
        self.fill = fill
        self.width = width
        self.last_update = 0
        self.sub_width = len(fill)

    def update(self, current):
        """
        Updates the progress bar with the current progress
        @param current: The current progress index
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time - self.last_update < 0.1:  # Update at most 10 times per second
            return
        self.last_update = elapsed_time
        percent = float(current) / self.total
        filled_length = int(round(percent * self.width * self.sub_width))
        sub_filled = filled_length % self.sub_width
        filled_blocks = filled_length // self.sub_width
        bar = self.fill[-1] * filled_blocks + self.fill[sub_filled] + '▁' * (self.width - filled_blocks - 1)
        progress_str = "{0}/{1}".format(current, self.total)
        full_bar = "{0} {1} {2} {3}".format(self.prefix, bar, progress_str, self.suffix)
        sys.stdout.write('\r' + full_bar)
        sys.stdout.flush()

    def __iter__(self):
        """
        Iterates over the iterable and updates the progress bar for each iteration
        """
        self.start_time = time.time()
        self.last_update = 0
        for i, item in enumerate(self.iterable):
            self.update(i + 1)
            yield item


def spinner_animation():
    """
    A function that displays a spinner animation while a function is running
    """
    spinner = '|/-\\'
    while not threading.current_thread().stopped:
        for s in spinner:
            sys.stdout.write('\r' + s)
            sys.stdout.flush()
            time.sleep(0.1)

def run_with_spinner(func):
    """
    Runs a function with a spinner animation
    @param func: The function to run
    """
    # Create a thread to display the spinner
    spinner_thread = threading.Thread(target=spinner_animation)
    spinner_thread.stopped = False
    spinner_thread.start()

    # Run the function and stop the spinner thread when it's done
    try:
        func()
    except KeyboardInterrupt as e:
        pass
    finally:
        spinner_thread.stopped = True
        spinner_thread.join()

if __name__ == "__main__":
# Example usage
    from time import sleep

# Define the range to iterate over
    range_to_iterate = range(200)

# Create a progress bar instance with a prefix and suffix
    pb = ProgressBar(range_to_iterate, prefix='Generating embeddings:', suffix='Complete', width=50)

# Iterate over the range and sleep for a bit to simulate work
    for i in pb:
        sleep(0.05)

# Print a newline at the end to separate the progress bar from the next output
    print('\nDone!\n')
    def long_running_function():
            """
            A function that takes a long time to run
            """
            time.sleep(3)

    run_with_spinner(long_running_function)

