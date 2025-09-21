import os
import sys
import subprocess
import time
from datetime import datetime
import threading
import itertools
import argparse

class Spinner:
    def __init__(self, message="Processing"):
        self.spinner = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
        self.running = False
        self.message = message
        self.thread = None

    def spin(self):
        while self.running:
            sys.stdout.write(f"\r{self.message} {next(self.spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2))

    def start(self, message=None):
        if message:
            self.message = message
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write('\r')

def print_banner():
    banner = """
##########################################################################
#                                                                        #
#                === Quantitative Trading Pipeline ===                   #
#                                                                        #
##########################################################################

            ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗
           ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝
           ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   
           ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   
           ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   
            ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   

           ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗
           ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝
              ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗
              ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
              ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
              ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 

##########################################################################
#                                                                        #
#                    === Interactive Pipeline Mode ===                   #
#                                                                        #
##########################################################################
"""
    print(banner)

def get_user_input():
    """Get user input with validation."""
    hypothesis = input("Enter the hypothesis (e.g., hyp_a): ").strip()
    
    # For symbol, use XAUUSD as default if nothing is entered
    symbol = input("Enter the symbol (default: XAUUSD): ").strip()
    if not symbol:
        symbol = "XAUUSD"
    
    while True:
        try:
            year = int(input("Enter the start year (e.g., 2018): ").strip())
            if year < 2015:
                print("Error: Start year cannot be before 2015.")
                continue
            break
        except ValueError:
            print("Please enter a valid year (e.g., 2018)")
    
    # For timeframe, use H1 as default if nothing is entered
    timeframe = input("Enter the timeframe (default: H1): ").strip()
    if not timeframe:
        timeframe = "H1"
    # Convert timeframe to uppercase
    timeframe = timeframe.upper()
    
    return hypothesis, symbol, year, timeframe

def validate_inputs(hypothesis, symbol, year, timeframe):
    """Validate user inputs before running the pipeline."""
    if not hypothesis:
        print("Error: Hypothesis cannot be empty")
        return False
    
    # Define valid timeframes
    valid_timeframes = {
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30',
        'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1'
    }
    
    if timeframe not in valid_timeframes:
        print(f"Error: Invalid timeframe. Please use one of: {', '.join(sorted(valid_timeframes))}")
        return False
        
    return True

def run_step(step_name, script_name, args):
    """Run a pipeline step and handle any errors."""
    print("\n" + "="*50)
    print(f"{step_name:^50}")
    print("="*50)
    print(f"\nExecuting: {script_name}")
    print(f"Arguments: {' '.join(args)}\n")
    
    try:
        # Run the command
        cmd = [sys.executable, script_name] + args
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            # Check if process has finished
            if process.poll() is not None:
                break
        
        # Get the remaining output
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout.strip())
        if stderr:
            print("\nWarnings/Errors:", file=sys.stderr)
            print(stderr.strip(), file=sys.stderr)
        
        if process.returncode != 0:
            print(f"\nError: {script_name} failed with return code {process.returncode}")
            return False
            
        print(f"\n{step_name} completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nUnexpected error running {script_name}: {e}")
        return False

def main():
    # Clear screen and show banner
    os.system('cls' if os.name == 'nt' else 'clear')
    print_banner()
    
    # Get user inputs
    hypothesis, symbol, year, timeframe = get_user_input()
    
    # Validate inputs
    if not validate_inputs(hypothesis, symbol, year, timeframe):
        print("\nError: Invalid inputs. Please try again.")
        sys.exit(1)
    
    # Create necessary directories if they don't exist
    os.makedirs("../reports", exist_ok=True)
    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../reports/report_{hypothesis}_{symbol}_{year}_{timeframe}_{timestamp}.log"
    
    # Run pipeline steps
    steps = [
        ("DATA ACQUISITION", "data_acquisition.py", 
         ["--year", str(year)]),  # data_acquisition.py only needs year
        
        ("FEATURE ENGINEERING", f"{hypothesis}_feature_engineering.py",
         ["--year", str(year), "--timeframe", timeframe.lower(), "--symbol", symbol.lower()]),
        
        ("MODEL TRAINING", f"{hypothesis}_train_model.py",
         ["--year", str(year), "--hypothesis", hypothesis])
    ]
    
    success = True
    for step_name, script, args in steps:
        if not run_step(step_name, script, args):
            success = False
            break
    
    print("\n" + "="*39)
    print(f"{'PIPELINE COMPLETED':^39}")
    print("="*39)
    
    if not success:
        print("\nPipeline encountered errors. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()