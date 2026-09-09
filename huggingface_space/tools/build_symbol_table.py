"""Build a bundled symbol -> company-name table for the UI's autocomplete.

Network is used HERE, at build time, so the running Space needs none: a visitor who
knows "Nvidia" but not "NVDA" must not depend on a live lookup that can rate-limit,
lapse, or be offline. Names are fetched from yfinance and verified, then written to a
static JSON the app loads at import.
"""
import csv
import json
import sys
import warnings

warnings.filterwarnings("ignore")
import yfinance as yf

OUT = "C:/Users/chaitanyakharche/Documents/stock/huggingface_space/trade_analysis/symbols.json"
QQQ_CSV = "C:/Users/chaitanyakharche/Desktop/data/reference/qqq_constituents.csv"

# Broad, liquid, recognisable. QQQ constituents are merged in below.
EXTRA = """
SPY QQQ IWM DIA VTI VOO VEA VWO EFA EEM AGG BND TLT IEF SHY HYG LQD GLD SLV USO UNG
XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC SMH SOXX XBI IBB ARKK VNQ KRE ITB JETS
BRK-B JPM BAC WFC GS MS C SCHW BLK AXP V MA PYPL COIN HOOD SOFI
JNJ PFE MRK ABBV LLY UNH CVS CI HUM TMO DHR ABT MDT SYK BSX ZTS
XOM CVX COP SLB EOG PSX MPC VLO OXY KMI WMB
WMT COST TGT HD LOW DG DLTR KR SYY MCD SBUX CMG YUM NKE LULU TJX ROST
PG KO PEP PM MO CL KMB GIS K HSY STZ MDLZ
BA CAT DE GE HON MMM UPS FDX LMT RTX NOC GD UNP CSX NSC WM
T VZ TMUS DIS CMCSA NFLX WBD PARA SPOT RBLX EA TTWO U
CRM ORCL SAP NOW SNOW DDOG MDB NET CRWD PANW ZS OKTA TEAM WDAY
UBER LYFT DASH ABNB BKNG MAR HLT RCL CCL NCLH DAL UAL AAL LUV
F GM RIVN LCID NIO STLA TM HMC
PLTR SNAP PINS SQ SHOP SPOT TWLO ZM DOCU ROKU
NEE DUK SO D AEP EXC SRE XEL
LIN APD SHW ECL NEM FCX NUE STLD X CLF AA
SMCI DELL HPQ HPE IBM ACN INFY CTSH
"""


def main() -> int:
    want = {}
    try:
        with open(QQQ_CSV, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("symbol"):
                    want[row["symbol"].strip().upper()] = (row.get("name") or "").strip()
    except OSError as exc:
        print(f"  QQQ constituents unavailable ({exc}); continuing with EXTRA only")
    for s in EXTRA.split():
        want.setdefault(s.upper(), "")

    print(f"  resolving {len(want)} symbols ...")
    out, missing = {}, []
    for i, (sym, seed_name) in enumerate(sorted(want.items()), 1):
        name = seed_name
        try:
            info = yf.Ticker(sym).get_info()
            name = (info.get("longName") or info.get("shortName") or seed_name or "").strip()
        except Exception:                                    # noqa: BLE001
            pass
        if name:
            out[sym] = name
        else:
            missing.append(sym)
        if i % 50 == 0:
            print(f"    {i}/{len(want)} ...")

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(sorted(out.items())), fh, indent=0, ensure_ascii=False)
    print(f"  wrote {len(out)} symbols to {OUT}")
    if missing:
        print(f"  no name resolved for {len(missing)}: {missing[:15]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
