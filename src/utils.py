from datetime import datetime
def generate_report(
    # output_path,
    hypothesis,
    year,
    symbol,
    timeframe,
    full_data,
    X_train,
    X_test,
    model_type,
    model_report,
    model_path
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_type.split(" ")[0] + "_" + model_type.split(" ")[-1]
    results_file = f'../reports/{hypothesis}_{symbol}_{timeframe}_{year}_{model_name.lower()}_results_{timestamp}.txt'
    
    results = []
    results.append("=== Model Training Results ===")
    results.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results.append(f"Hypothesis: {hypothesis}")
    results.append(f"Year: {year}")
    results.append(f"Symbol: {symbol}")
    results.append(f"Timeframe: {timeframe}")
    results.append(f"\nDataset Information:")
    results.append(f"Total samples: {str(full_data)}")
    results.append(f"Training samples: {str(X_train)}")
    results.append(f"Testing samples: {str(X_test)}")
    results.append(f"\n=== {model_type} Model Results ===")
    results.append(model_report)
    results.append("\n=== Model File Locations ===")
    results.append(f"Classification model: {model_path}")
    
    with open(results_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"\nResults saved to: {results_file}")
    