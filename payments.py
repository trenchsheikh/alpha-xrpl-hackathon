"""Organiser-compatible synthetic payment projection; one economic edge per transaction."""
import csv
import json
from decimal import Decimal
from pathlib import Path
import pyarrow.parquet as pq
SCALE=1_000_000

def load_rates(directory):
 with (Path(directory)/"market_rates.csv").open() as stream:
  return {(int(r["TIMESTAMP"]),r["ASSET"]):int(r["PRICE_MICRO_USD"]) for r in csv.DictReader(stream)}

def decode_amount(wire):
 if isinstance(wire,str): return "XRP",int(wire)
 return f"{wire['issuer']}:{wire['currency']}",int(Decimal(wire["value"])*SCALE)

def project(transactions,rates):
 rows=[]
 for tx in transactions:
  fields=json.loads(tx["EXTRA_TRANSACTION_FIELDS"])
  asset,units=decode_amount(json.loads(tx["DELIVERED_AMOUNT"]))
  usd=units*rates[(int(tx["TIMESTAMP"])//3600*3600,asset)]/(SCALE*SCALE)
  rows.append({"TRANSACTION_HASH":tx["TRANSACTION_HASH"],"TIMESTAMP":tx["TIMESTAMP"],"FROM_ADDRESS":tx["ACCOUNT"],"TO_ADDRESS":fields["Destination"],"VALUE":usd,"ASSET":asset})
 return rows

def load_payments(directory):
 directory=Path(directory)
 return project(pq.read_table(directory/"transactions.parquet").to_pylist(),load_rates(directory))
