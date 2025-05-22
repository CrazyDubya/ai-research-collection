import random
import pandas as pd
import concurrent.futures
import uuid

# Function to run trouoin.py and collect results
def run_trouoin():
    from trouoin import run_trouoin
    return run_trouoin()

# Run trouoin.py concurrently and collect results
results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_trouoin) for _ in range(random.randint(5, 9))]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())

# Concatenate all results into a single DataFrame
df_all = pd.concat(results, ignore_index=True)

# Print the DataFrame
print(df_all)