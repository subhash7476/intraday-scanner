# Add this function at the end
def run_scanner():
    # ... [same code] ...
    if results:
        final = pd.DataFrame(results).sort_values("Score", ascending=False).head(10)
        final.to_csv("intraday_signals_12.csv", index=False)
        return final
    return pd.DataFrame()
