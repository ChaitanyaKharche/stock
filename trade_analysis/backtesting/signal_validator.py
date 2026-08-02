import os
import re
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ..paths import LOGS_DIR

API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    sys.exit(1)

class SignalValidator:
    def __init__(self, log_file=None):
        self.log_file = Path(log_file) if log_file else LOGS_DIR / "strategy_trader_log.txt"
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        self.tz = ZoneInfo("America/New_York")
        
        self.entries = []
        self.exits = []
        self.levels = {} # { '2025-11-14': {'SPY': [], 'QQQ': []} }
        
        print(f"\n{'='*70}")
        print(f"SIGNAL VALIDATOR (v2) - Post-Mortem Analysis")
        print(f"{'='*70}")
        print(f"Log file: {log_file}")
    
    def parse_log(self):
        if not self.log_file.exists():
            print(f"ERROR: Log file not found: {self.log_file}")
            return False
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"\nParsing {len(lines)} log lines...")
        
        pending_entry = None
        pending_exit = None

        for line in lines:
            timestamp_match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
            
            if not timestamp_match:
                # Non-timestamped line, e.g., '===='
                if pending_entry:
                    # '====' line signifies the END of the entry block
                    if '====' in line:
                        if 'symbol' in pending_entry and 'entry_price' in pending_entry:
                            self.entries.append(pending_entry)
                        pending_entry = None
                
                if pending_exit:
                     # '====' line signifies the END of the exit block
                    if '====' in line:
                        if 'symbol' in pending_exit and 'exit_price' in pending_exit:
                            self.exits.append(pending_exit)
                        pending_exit = None
                continue

            # --- We have a timestamped line ---
            timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
            timestamp = timestamp.replace(tzinfo=self.tz)
            date_str = timestamp.date()

            # --- Is it an ENTRY? ---
            entry_match = re.search(r'🚀 \[(.+) (CALL|PUT) ENTRY\]', line)
            if entry_match:
                pending_entry = {
                    'time': timestamp,
                    'strategy': entry_match.group(1),
                    'type': entry_match.group(2),
                    'level_name': 'Unknown'
                }
                continue # Go to next line for details

            # --- Is it an EXIT? ---
            exit_match = re.search(r'🛑 \[(TARGET|STOP) EXIT\] \((.+)\)', line)
            if exit_match:
                pending_exit = {
                    'time': timestamp,
                    'type': exit_match.group(1), # TARGET or STOP
                    'strategy': exit_match.group(2)
                }
                continue # Go to next line for details

            # --- Is it an Entry Detail line? ---
            if pending_entry:
                symbol_match = re.search(r'Symbol:\s+(SPY|QQQ|TSLA)\s+@\s+\$(\d+\.?\d*)', line)
                trigger_match = re.search(r'Trigger:\s+(.+)', line)
                
                if symbol_match:
                    pending_entry['symbol'] = symbol_match.group(1)
                    pending_entry['entry_price'] = float(symbol_match.group(2))
                if trigger_match:
                    pending_entry['level_name'] = trigger_match.group(1).strip()
                continue
            
            # --- Is it an Exit Detail line? ---
            if pending_exit:
                symbol_match = re.search(r'Symbol:\s+(SPY|QQQ|TSLA)\s+@\s+\$(\d+\.?\d*)', line)
                pnl_match = re.search(r'P&L:\s+\$([+-]?\d+\.?\d*)', line) 

                if symbol_match:
                    pending_exit['symbol'] = symbol_match.group(1)
                    pending_exit['exit_price'] = float(symbol_match.group(2))
                if pnl_match:
                    pending_exit['pnl_dollars'] = float(pnl_match.group(1))
                continue
            
            # --- Is it a Level Definition line? ---
            level_match = re.search(r'(SUP|RES):\s+(.+?)\s+@\s+(\d+\.?\d*)', line)
            orb_match = re.search(r'(SPY|QQQ|TSLA)\s+ORB:\s+High\s+\$(\d+\.?\d*),\s+Low\s+\$(\d+\.?\d*)', line)

            if date_str not in self.levels:
                self.levels[date_str] = {'SPY': [], 'QQQ': [], 'TSLA': []}

            if level_match:
                try:
                    symbol = 'SPY' if 'SPY' in lines[lines.index(line) - 1] else \
                             'QQQ' if 'QQQ' in lines[lines.index(line) - 1] else 'TSLA'
                    level_name = level_match.group(2).strip()
                    level_price = float(level_match.group(3))
                    self.levels[date_str][symbol].append((level_name, level_price))
                except:
                    pass # Failed to find symbol
            
            if orb_match:
                symbol = orb_match.group(1)
                orb_high = float(orb_match.group(2))
                orb_low = float(orb_match.group(3))
                self.levels[date_str][symbol].append((f"ORB High", orb_high))
                self.levels[date_str][symbol].append((f"ORB Low", orb_low))

        print(f"\n✅ Parsed:")
        print(f"  Entries: {len(self.entries)}")
        print(f"  Exits: {len(self.exits)}")
        
        return True
    
    def fetch_price_data(self, symbol, start_time, end_time):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time,
                feed="sip"
            )
            
            bars = self.data_client.get_stock_bars(request)
            if bars is None or bars.df.empty:
                return None
            
            df = bars.df
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df = df[df['symbol'] == symbol].copy()
                df.set_index('timestamp', inplace=True)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert(self.tz)
            
            return df
        
        except Exception as e:
            print(f"ERROR fetching {symbol}: {str(e)[:80]}")
            return None
    
    def create_chart(self, symbol, date):
        print(f"\n📊 Creating chart for {symbol} on {date.strftime('%Y-%m-%d')}...")
        
        # --- Filter events for THIS date ---
        symbol_entries = [e for e in self.entries if e['symbol'] == symbol and e['time'].date() == date]
        symbol_exits = [e for e in self.exits if e['symbol'] == symbol and e['time'].date() == date]
        symbol_levels = self.levels.get(date, {}).get(symbol, [])
        
        if not symbol_entries and not symbol_exits:
            print(f"  No trades for {symbol} on this date.")
            return None
        
        start = datetime.combine(date, time(9, 30), tzinfo=self.tz)
        end = datetime.combine(date, time(16, 1), tzinfo=self.tz) 
        
        df = self.fetch_price_data(symbol, start, end)
        if df is None or df.empty:
            print(f"  No price data available for {symbol} on {date.strftime('%Y-%m-%d')}")
            df = pd.DataFrame(index=pd.date_range(start, end, freq='1min'))
            df['open'] = None; df['high'] = None; df['low'] = None; df['close'] = None; df['volume'] = 0
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} - Signal Analysis', 'Volume')
        )
        
        # Only add candles if we have data
        if df['open'].notnull().any():
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
            
            colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' 
                    for i in range(len(df)) if pd.notnull(df['close'].iloc[i]) and pd.notnull(df['open'].iloc[i])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index, y=df['volume'], name='Volume',
                    marker_color=colors, showlegend=False
                ),
                row=2, col=1
            )
        
        # --- Plot Levels ---
        unique_levels = {}
        for level_name, level_price in symbol_levels:
             if level_name not in unique_levels:
                unique_levels[level_name] = level_price

        for level_name, level_price in unique_levels.items():
            color = '#FF9800' # Default orange
            if 'High' in level_name: color = '#2196F3' # Blue
            if 'Low' in level_name: color = '#FFC107' # Amber
            
            fig.add_hline(
                y=level_price, line_dash="dash", line_color=color, line_width=1,
                opacity=0.7,
                annotation_text=level_name,
                annotation_position="right",
                row=1, col=1
            )
        
        # --- Plot Entries ---
        for e in symbol_entries:
            color = '#D50000' if e['type'] == 'PUT' else '#00C853'
            symbol_icon = 'triangle-down' if e['type'] == 'PUT' else 'triangle-up'
            fig.add_trace(
                go.Scatter(
                    x=[e['time']], y=[e['entry_price']],
                    mode='markers+text',
                    marker=dict(symbol=symbol_icon, size=18, color=color, line=dict(color='white', width=2)),
                    text=['ENTRY'],
                    textposition='top center' if e['type'] == 'CALL' else 'bottom center',
                    textfont=dict(size=12, color='white'),
                    name=f"{e['type']} Entry ({e['strategy']})",
                    hovertemplate=f"<b>{e['strategy']} {e['type']}</b><br>{e['level_name']}<br>Time: %{{x}}<br>Entry: ${e['entry_price']:.2f}<extra></extra>",
                    showlegend=True
                ),
                row=1, col=1
            )
        
        exit_colors = {'TARGET': '#4CAF50', 'STOP': '#F44336'}
        
        # --- Plot Exits ---
        for ex in symbol_exits:
            fig.add_trace(
                go.Scatter(
                    x=[ex['time']], y=[ex['exit_price']],
                    mode='markers+text',
                    marker=dict(symbol='x', size=15, color=exit_colors.get(ex['type'], 'grey'), line=dict(color='black', width=2)),
                    text=[ex['type']],
                    textposition='bottom center',
                    textfont=dict(size=10),
                    name=f"{ex['type']} Exit",
                    hovertemplate=f"<b>{ex['type']} EXIT</b><br>Strategy: {ex['strategy']}<br>Time: %{{x}}<br>Exit: ${ex['exit_price']:.2f}<br>P&L: ${ex['pnl_dollars']:+.2f}<extra></extra>",
                    showlegend=True
                ),
                row=1, col=1
            )
        
        fig.update_layout(
            title=f"{symbol} - {date.strftime('%Y-%m-%d')} Signal Analysis",
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark',
            hovermode='x unified',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            dragmode='pan'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1, range=[start, end])
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig

    def generate_report(self, output_file=None):
        output_file = output_file or str(LOGS_DIR / "signal_validation_report.html")
        if not self.parse_log():
            return
        
        all_events = self.entries + self.exits
        if not all_events:
            print("No events found in log!")
            return
        
        all_dates = sorted(list(set([e['time'].date() for e in all_events])))
        all_symbols = sorted(list(set([e['symbol'] for e in all_events])))
        
        print(f"\n📈 Generating report for {len(all_dates)} days and {len(all_symbols)} symbols...")
        
        html_parts = [f"""
        <html>
        <head>
            <title>Signal Validation Report</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #0a0e27; color: #fff; margin: 0; padding: 20px; }}
                h1 {{ text-align: center; color: #00ff88; margin-bottom: 10px; }}
                h2 {{ color: #00aaff; border-bottom: 2px solid #00aaff; padding-bottom: 5px; margin-top: 40px; }}
                h3 {{ color: #eee; margin-top: 30px; }}
                .summary {{ background: #1a1f3a; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .stat {{ display: inline-block; margin: 10px 20px; font-size: 18px; }}
                .stat-value {{ color: #00ff88; font-weight: bold; font-size: 24px; }}
                .chart-container {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>📊 Signal Validation Report</h1>
            
            <div class="summary">
                <h2>📈 Summary (All Days)</h2>
                <div class="stat">
                    <span class="stat-value">{len(self.entries)}</span><br>
                    Entries Signaled
                </div>
                <div class="stat">
                    <span class="stat-value">{len(self.exits)}</span><br>
                    Exits Executed
                </div>
            </div>
        """]
        
        for date in all_dates:
            html_parts.append(f"<h2>Charts for {date.strftime('%A, %B %d, %Y')}</h2>")
            has_chart_for_date = False
            for symbol in all_symbols:
                fig = self.create_chart(symbol, date)
                if fig:
                    has_chart_for_date = True
                    html_parts.append(f'<div class="chart-container"><h3>{symbol}</h3>')
                    html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn', config={
                        'displayModeBar': True, 'displaylogo': False,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                        'scrollZoom': True
                    }))
                    html_parts.append('</div>')
            
            if not has_chart_for_date:
                 html_parts.append(f"<p>No chartable events for this day.</p>")
        
        html_parts.append("""
            <div style="text-align: center; margin-top: 50px; color: #666;">
                <p>Generated by Signal Validator</p>
            </div>
        </body>
        </html>
        """)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
        
        print(f"\n✅ Report generated: {output_file}")
        print(f"   Open in browser to review signals visually")
        
        print(f"\n{'='*70}")
        print(f"KEY INSIGHTS (All Days):")
        print(f"{'='*70}")
        
        if self.entries:
            print(f"\n📍 Entries:")
            for e in sorted(self.entries, key=lambda x: x['time']):
                print(f"  {e['time'].strftime('%Y-%m-%d %H:%M')} | {e['symbol']} {e['type']} ({e['strategy']}) @ ${e['entry_price']:.2f}")
                print(f"     Trigger: {e['level_name']}")
        
        if self.exits:
            print(f"\n🎯 Exits:")
            total_pnl = 0
            wins = 0
            losses = 0
            for ex in sorted(self.exits, key=lambda x: x['time']):
                pnl_dollars = ex.get('pnl_dollars', 0)
                total_pnl += pnl_dollars
                if ex['type'] == 'TARGET':
                    wins += 1
                    print(f"  {ex['time'].strftime('%Y-%m-%d %H:%M')} | {ex['symbol']} {ex['type']} @ ${ex['exit_price']:.2f} (P&L: ${pnl_dollars:+.2f})")
                else: # STOP
                    losses += 1
                    print(f"  {ex['time'].strftime('%Y-%m-%d %H:%M')} | {ex['symbol']} {ex['type']} @ ${ex['exit_price']:.2f} (P&L: ${pnl_dollars:+.2f})")

            print(f"     ---")
            print(f"     Wins: {wins} | Losses: {losses} | Total P&L: ${total_pnl:+.2f}")

def main():
    validator = SignalValidator()
    validator.generate_report()

if __name__ == "__main__":
    main()