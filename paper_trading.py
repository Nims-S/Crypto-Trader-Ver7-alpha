"""
Paper trading stage (stub).

Uses backtest engine on recent data as forward proxy.
"""

from backtest import run_backtest
from strategy_registry import get_strategy, record_experiment

def paper_trade(strategy_id, symbol, timeframe, bars=300):
    strat = get_strategy(strategy_id)

    if not strat:
        print("Strategy not found")
        return

    result = run_backtest(
        symbol,
        timeframe,
        max_bars=bars,
        strategy_override={
            "strategy_id": strat["strategy_id"],
            "parameters": strat["parameters"]
        }
    )

    record_experiment(
        strategy_id,
        symbol=symbol,
        timeframe=timeframe,
        run_type="paper_forward",
        parameters=strat["parameters"],
        metrics=result,
        passed=result.get("profit_factor", 0) > 1.0,
        notes="paper stage stub"
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="4h")
    args = parser.parse_args()

    res = paper_trade(args.strategy, args.symbol, args.timeframe)
    print(res)