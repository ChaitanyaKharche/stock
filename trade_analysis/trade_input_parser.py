"""
Trade Input Parser
Converts simple trade logs to the format needed by trade_journal_analyzer.py

Create a trades_input.csv file with your trades, then run this script.
"""

import csv
import re
from datetime import datetime, timedelta
from dateutil import parser as date_parser

# ============================================================================
# PASTE YOUR TRADES HERE IN SIMPLE FORMAT
# ============================================================================

# Format: One trade per line
# "Action Symbol $Strike Type Expiry | Contracts @ Price | Date/Time | Notes"
# 
# Examples:
# Buy QQQ $627 Call 12/4 | 3 @ $0.68 | Dec 3 14:00 | afternoon scalp
# Sell QQQ $627 Call 12/4 | 3 @ $0.41 | Dec 3 15:30 | stop loss

RAW_TRADES = """
Sell QQQ $627 Call 12/4 | 3 @ $0.41 | Dec 3 15:30
Buy QQQ $627 Call 12/4 | 3 @ $0.68 | Dec 3 14:00
Sell QQQ $625 Call 12/3 | 5 @ $0.19 | Dec 3 12:00
Buy QQQ $625 Call 12/3 | 5 @ $0.36 | Dec 3 10:15
Sell QQQ $615 Put 12/2 | 3 @ $0.30 | Dec 2 15:00
Buy QQQ $615 Put 12/2 | 3 @ $0.64 | Dec 2 13:30
Sell QQQ $627 Call 12/3 | 4 @ $0.50 | Dec 2 13:00
Buy QQQ $627 Call 12/3 | 4 @ $0.85 | Dec 2 10:30
Sell QQQ $625 Call 12/2 | 4 @ $0.27 | Dec 2 11:30
Buy QQQ $625 Call 12/2 | 4 @ $0.60 | Dec 2 10:00
Sell QQQ $625 Call 12/2 | 3 @ $0.17 | Dec 1 14:30
Buy QQQ $625 Call 12/2 | 3 @ $0.39 | Dec 1 11:00
Sell QQQ $620 Call 12/2 | 1 @ $1.79 | Dec 1 14:00
Sell QQQ $620 Call 12/2 | 1 @ $1.59 | Dec 1 12:00
Sell QQQ $620 Call 12/2 | 2 @ $1.06 | Dec 1 10:30
Buy QQQ $620 Call 12/2 | 4 @ $1.24 | Dec 1 09:45
"""


def parse_raw_trade(line, year=2024):
    """Parse a single trade line"""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    trade = {}
    
    # Split by pipe
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 3:
        print(f"[WARN] Could not parse: {line}")
        return None
    
    option_part = parts[0]
    qty_part = parts[1]
    time_part = parts[2]
    notes_part = parts[3] if len(parts) > 3 else ''
    
    # Parse option part: "Buy QQQ $627 Call 12/4"
    match = re.match(
        r'(Buy|Sell)\s+(\w+)\s+\$?([\d.]+)\s+(Call|Put)\s+(\d+/\d+)',
        option_part,
        re.IGNORECASE
    )
    
    if not match:
        print(f"[WARN] Could not parse option part: {option_part}")
        return None
    
    trade['action'] = match.group(1).upper()
    trade['symbol'] = match.group(2).upper()
    trade['underlying'] = trade['symbol']
    trade['strike'] = float(match.group(3))
    trade['option_type'] = match.group(4).upper()
    trade['expiry'] = match.group(5)
    
    # Parse qty part: "3 @ $0.68"
    match = re.match(r'(\d+)\s*@\s*\$?([\d.]+)', qty_part)
    if match:
        trade['contracts'] = int(match.group(1))
        trade['price_per_contract'] = float(match.group(2))
    else:
        print(f"[WARN] Could not parse qty part: {qty_part}")
        return None
    
    # Parse time part: "Dec 3 14:00" or "Dec 3" or "2024-12-03 14:00:00"
    time_str = time_part.strip()
    
    try:
        # Add year if not present
        if str(year) not in time_str:
            time_str = f"{time_str} {year}"
        
        dt = date_parser.parse(time_str)
        
        # If no time specified, default to 10:00 AM
        if dt.hour == 0 and dt.minute == 0:
            dt = dt.replace(hour=10, minute=0)
        
        trade['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"[WARN] Could not parse time: {time_str} - {e}")
        return None
    
    trade['notes'] = notes_part
    
    return trade


def parse_all_trades(raw_text, year=2024):
    """Parse all trades from raw text"""
    trades = []
    lines = raw_text.strip().split('\n')
    
    for line in lines:
        trade = parse_raw_trade(line, year)
        if trade:
            trades.append(trade)
    
    return trades


def export_trades_python(trades, filename='trades_parsed.py'):
    """Export trades as Python list for use in analyzer"""
    with open(filename, 'w') as f:
        f.write("# Auto-generated trade list\n")
        f.write("# Copy this into trade_journal_analyzer.py TRADES list\n\n")
        f.write("TRADES = [\n")
        
        for trade in trades:
            f.write("    {\n")
            for k, v in trade.items():
                if isinstance(v, str):
                    f.write(f"        '{k}': '{v}',\n")
                else:
                    f.write(f"        '{k}': {v},\n")
            f.write("    },\n")
        
        f.write("]\n")
    
    print(f"Exported {len(trades)} trades to {filename}")


def export_trades_csv(trades, filename='trades_parsed.csv'):
    """Export trades as CSV"""
    if not trades:
        return
    
    keys = trades[0].keys()
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(trades)
    
    print(f"Exported {len(trades)} trades to {filename}")


def print_trades_summary(trades):
    """Print summary of parsed trades"""
    print("\n" + "="*60)
    print("PARSED TRADES SUMMARY")
    print("="*60)
    
    total_buys = sum(1 for t in trades if t['action'] == 'BUY')
    total_sells = sum(1 for t in trades if t['action'] == 'SELL')
    
    print(f"Total trades: {len(trades)}")
    print(f"  Buys: {total_buys}")
    print(f"  Sells: {total_sells}")
    
    # Group by date
    dates = {}
    for t in trades:
        date = t['timestamp'][:10]
        if date not in dates:
            dates[date] = []
        dates[date].append(t)
    
    print(f"\nTrades by date:")
    for date in sorted(dates.keys()):
        print(f"  {date}: {len(dates[date])} trades")
    
    # Print each trade
    print("\n" + "-"*60)
    print("TRADE LIST:")
    print("-"*60)
    
    for i, t in enumerate(trades, 1):
        total = t['contracts'] * t['price_per_contract'] * 100
        print(f"{i:2}. {t['timestamp'][:16]} | {t['action']:4} {t['contracts']}x {t['symbol']} ${t['strike']} {t['option_type']} | ${total:.0f}")


if __name__ == "__main__":
    print("="*60)
    print("TRADE INPUT PARSER")
    print("="*60)
    
    # Parse trades
    trades = parse_all_trades(RAW_TRADES, year=2024)
    
    if trades:
        print_trades_summary(trades)
        
        # Export
        export_trades_python(trades)
        export_trades_csv(trades)
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Copy contents of trades_parsed.py into trade_journal_analyzer.py")
        print("2. Run: python trade_journal_analyzer.py")
        print("3. Check the output CSV for your analysis")
    else:
        print("No trades parsed. Check your input format.")
