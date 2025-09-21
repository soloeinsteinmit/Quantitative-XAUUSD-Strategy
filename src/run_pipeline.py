import os
import sys
import subprocess
import time
from datetime import datetime

import argparse



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
    """Get user input with validation and retry options."""
    while True:
        try:
            # Hypothesis is mandatory
            while True:
                hypothesis = input("Enter the hypothesis (e.g., hyp_a): ").strip().lower()
                if hypothesis:
                    break
                print("Error: Hypothesis is required. Please enter a valid hypothesis.")
            
            # Symbol with default
            symbol = input("Enter the symbol (default: XAUUSD): ").strip().upper()
            symbol = symbol if symbol else "XAUUSD"
            
            # Year with default and validation
            year_input = input("Enter the start year (default: 2018): ").strip()
            if not year_input:
                year = 2018
            else:
                try:
                    year = int(year_input)
                    if year < 2015:
                        print("Error: Start year cannot be before 2015.")
                        continue
                except ValueError:
                    print("Error: Please enter a valid year (e.g., 2018)")
                    continue
            
            # Timeframe with default
            timeframe = input("Enter the timeframe (default: H1): ").strip().upper()
            timeframe = timeframe if timeframe else "H1"
            
            # Final validation
            print("\nConfirm your selections:")
            print(f"Hypothesis: {hypothesis}")
            print(f"Symbol: {symbol}")
            print(f"Year: {year}")
            print(f"Timeframe: {timeframe}")
            
            confirm = input("\nIs this correct? (Y/n): ").strip().lower()
            if confirm == '' or confirm == 'y':
                return hypothesis, symbol, year, timeframe
            print("\nLet's try again...\n")
            
        except KeyboardInterrupt:
            print("\n\nInput cancelled. Would you like to:")
            print("1. Try again")
            print("2. Exit")
            try:
                choice = input("Enter your choice (1/2): ").strip()
                if choice == "2":
                    sys.exit(0)
                print("\nLet's try again...\n")
            except KeyboardInterrupt:
                sys.exit(0)

def check_data_exists(symbol, timeframe, year):
    """Check if data file already exists."""
    filename = f"{symbol.lower()}_{timeframe.lower()}_{year}_present.parquet"
    filepath = os.path.join("..", "data", "raw", filename)
    return os.path.exists(filepath)

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
    
    # Check if data already exists
    if check_data_exists(symbol, timeframe, year):
        print(f"\nNote: Data file for {symbol}_{timeframe}_{year} already exists.")
        while True:
            try:
                redownload = input("Would you like to re-download the data? (y/N): ").strip().lower()
                if redownload == 'y':
                    return True
                elif redownload == '' or redownload == 'n':
                    print("Using existing data file...")
                    return True
                print("Please enter 'y' for yes or 'n' for no.")
            except KeyboardInterrupt:
                return False
        
    return True

def run_step(step_name, script_name, args):
    """Run a pipeline step and handle any errors, ensuring it completes before returning."""
    print("\n" + "="*50)
    print(f"{step_name:^50}")
    print("="*50)
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nStarting: {script_name} at {timestamp_start}")
    print(f"Arguments: {' '.join(args)}\n")
    
    try:
        # Use subprocess.run to execute the command and wait for it to complete
        process = subprocess.run(
            [sys.executable, script_name] + args,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print the standard output from the completed process
        print("\n--- Script Output ---")
        for line in process.stdout.splitlines():
            print(line)
        
        # If there were any warnings, print them
        if process.stderr:
            print("\n--- Warnings/Errors ---", file=sys.stderr)
            print(process.stderr, file=sys.stderr)
            
        timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{step_name} completed at {timestamp_end}")
        return True
        
    except subprocess.CalledProcessError as e:
        # This block runs if the script returns a non-zero exit code
        print(f"\nError: {script_name} failed with return code {e.returncode}", file=sys.stderr)
        print("\n--- Script Output (stdout) ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- Script Error (stderr) ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return False
        
    except Exception as e:
        print(f"\nAn unexpected error occurred while running {script_name}: {e}", file=sys.stderr)
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
    
    # Create necessary directories
    try:
        os.makedirs("../reports", exist_ok=True)
        os.makedirs("../data/raw", exist_ok=True)
        os.makedirs("../data/processed", exist_ok=True)
        os.makedirs("../models", exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)
    
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