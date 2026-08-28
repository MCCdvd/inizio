# Short Selling Support

## Overview

The trading agent now supports **short selling** in addition to traditional long-only trading.

## Action Types

The agent has 3 actions available:

### **Action 0: HOLD**
- Do nothing, keep current position
- Applied to both long and short positions

### **Action 1: BUY or COVER**
When `shares_held >= 0` (neutral or long):
- **BUY**: Open a long position with available capital
- Profit when price goes UP

When `shares_held < 0` (short):
- **COVER**: Close short position by buying back shares
- Reduces short position size
- Profit realized if price fell since short entry

### **Action 2: SELL or SHORT SELL**
When `shares_held > 0` (long):
- **SELL**: Close long position
- Profit when original entry price < current price

When `shares_held <= 0` (neutral or short):
- **SHORT SELL**: Open short position (sell without owning)
- Profit when price goes DOWN
- `shares_held` becomes negative (e.g., -100 shares)

### **Liquidation**
At episode end:
- **Long**: All shares are sold at final price
- **Short**: All shares are bought back at final price

## Position Tracking

**`shares_held` field:**
- Positive value: Long position (owns shares)
- Negative value: Short position (owes shares)
- Zero: No position

**Example:**
```python
shares_held = 100   # Long: owns 100 shares
shares_held = -50   # Short: owes 50 shares
shares_held = 0     # Neutral: no position
```

## State Representation

The `position_size` feature in the state vector can now be negative:

```python
position_size = shares_held / 100
# Can range from negative to positive
# Agent learns: negative = short, positive = long
```

## Profit Calculation

### Long Trade
```
Profit = (Sell Price - Buy Price) × Shares
```

### Short Trade
```
Profit = (Short Entry Price - Cover Price) × Shares
```

Both types profit when their directional thesis is correct.

## Risk Management

Short positions have built-in constraints:
1. **Collateral requirement**: Max short size is limited by available capital
2. **Fee cost**: Transaction fees apply to both opening and closing shorts
3. **Holding penalty**: Short positions held >3 days without sufficient downside incur penalties
4. **Forced liquidation**: All positions (long/short) are liquidated at episode end

## Example Scenarios

### Scenario 1: Successful Long Trade
```
1. Action 1 (BUY) at $100 → shares_held = 100
2. Hold
3. Action 2 (SELL) at $110 → shares_held = 0
Result: +$1,000 profit
```

### Scenario 2: Successful Short Trade
```
1. Action 2 (SHORT) at $110 → shares_held = -100
2. Hold
3. Action 1 (COVER) at $100 → shares_held = 0
Result: +$1,000 profit
```

### Scenario 3: Mixed Strategy
```
1. Action 1 (BUY) at $100 → shares_held = 100
2. Hold
3. Action 2 (SELL) at $105 → shares_held = 0 (profit: $500)
4. Action 2 (SHORT) at $105 → shares_held = -50
5. Action 1 (COVER) at $100 → shares_held = 0 (profit: $250)
Total profit: $750
```

## Training Benefits

With short selling enabled:
- ✅ Agent can profit in **downtrending markets**
- ✅ Better **risk-adjusted returns** (Sharpe ratio)
- ✅ Hedging opportunities
- ✅ More **diverse trading strategies**
- ✅ Can exploit **technical resistance levels** with shorts

## Backward Compatibility

- ✅ Existing long-only strategies still work
- ✅ Agents can learn to avoid shorts if not profitable
- ✅ No changes to long-only trade logic
- ✅ State space is same size (6 dimensions)

## Testing

To verify short selling works correctly:
```bash
cd trading-agent
PYTHONPATH=src python -m train --algorithm dqn --stock AAPL --episodes 250
```

Check output for:
- `SHORT` trades in trades.csv
- `COVER` trades in trades.csv
- Negative final portfolio values (if shorts dominate)
- Profit factors > 1.0 for both long and short

## Configuration

No special configuration needed! Short selling is automatically available.

The agent learns whether to use it based on reward signal:
- Profitable shorts → agent learns to short more
- Unprofitable shorts → agent learns to avoid them
