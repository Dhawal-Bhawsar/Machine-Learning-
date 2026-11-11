from pathlib import Path
import pandas as pd

parts = sorted(Path('.').glob('new_data_part_*.csv.gz'))
if not parts:
    raise FileNotFoundError('No new_data_part_*.csv.gz files found in this folder.')

frames = [pd.read_csv(part, compression='gzip') for part in parts]
df = pd.concat(frames, ignore_index=True)
df.to_csv('new_data.csv', index=False)
print(f'Reassembled new_data.csv with {len(df):,} rows and {len(df.columns)} columns.')
