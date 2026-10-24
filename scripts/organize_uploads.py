"""Match explicitly uploaded files to the organiser's train/validation layout.
Original unsuffixed files are only candidate test files until confirmed.
"""
from pathlib import Path
import argparse
import shutil

MAPPING = {
 "train": {"transactions.parquet":"transactions(1).parquet","transfers.parquet":"transfers(1).parquet","blocks.parquet":"blocks(2).parquet","market_rates.csv":"market_rates(1).csv","labels.csv":"labels.csv"},
 "validation": {"transactions.parquet":"transactions(2).parquet","transfers.parquet":"transfers(2).parquet","blocks.parquet":"blocks(1).parquet","market_rates.csv":"market_rates(2).csv"},
 "test": {"transactions.parquet":"transactions.parquet","transfers.parquet":"transfers.parquet","blocks.parquet":"blocks.parquet","market_rates.csv":"market_rates.csv"},
}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--source",required=True); p.add_argument("--destination",default="data")
 a=p.parse_args(); src=Path(a.source);dst=Path(a.destination)
 for split,files in MAPPING.items():
  if not all((src/f).exists() for f in files.values()):
   print("Skipping",split,"missing", [f for f in files.values() if not (src/f).exists()]);continue
  for target,original in files.items():
   out=dst/split/target;out.parent.mkdir(parents=True,exist_ok=True)
   if out.resolve() != (src/original).resolve(): shutil.copy2(src/original,out)
  print("Prepared",split,"in",dst/split)
 print("NOTE: test/ is only a candidate split unless organiser's separate test package confirms it.")
if __name__=="__main__":main()
