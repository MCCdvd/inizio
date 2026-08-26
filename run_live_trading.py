import sys
sys.path.insert(0, 'trading-agent/src')

from trading_agent import TradingEnvironmentWithVolumeProfile
from ib_insync import IB, Stock, Order
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTradingAgent:
    """Live trading agent with IBKR order execution"""
    
    def __init__(self, symbol="AAPL", initial_balance=10000):
        self.symbol = symbol
        self.initial_balance = initial_balance
        
        # Initialize trading environment with IBKR
        self.env = TradingEnvironmentWithVolumeProfile(
            stock_symbol=symbol,
            initial_balance=initial_balance,
            data_source="ibkr",
            ibkr_host="127.0.0.1",
            ibkr_port=7497,
            ibkr_timeframe="1 day"
        )
        
        # Initialize IB connection for order execution
        self.ib = IB()
        self.connected = False
        
    def connect(self):
        """Connect to IB Gateway"""
        try:
            logger.info("🔌 Connecting to IB Gateway...")
            self.ib.connect("127.0.0.1", 7497, clientId=12)
            self.connected = True
            logger.info("✅ Connected to IB Gateway")
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IB Gateway"""
        if self.connected:
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")
    
    def place_buy_order(self, symbol: str, quantity: int):
        """Place a BUY order"""
        try:
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            
            order = Order()
            order.action = "BUY"
            order.totalQuantity = quantity
            order.orderType = "MKT"  # Market order
            
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"📈 BUY ORDER PLACED: {quantity} x {symbol}")
            logger.info(f"   Order ID: {trade.order.orderId}")
            logger.info(f"   Status: {trade.orderStatus.status}")
            
            return trade
            
        except Exception as e:
            logger.error(f"❌ BUY order failed: {e}")
            return None
    
    def place_sell_order(self, symbol: str, quantity: int):
        """Place a SELL order"""
        try:
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            
            order = Order()
            order.action = "SELL"
            order.totalQuantity = quantity
            order.orderType = "MKT"  # Market order
            
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"📉 SELL ORDER PLACED: {quantity} x {symbol}")
            logger.info(f"   Order ID: {trade.order.orderId}")
            logger.info(f"   Status: {trade.orderStatus.status}")
            
            return trade
            
        except Exception as e:
            logger.error(f"❌ SELL order failed: {e}")
            return None
    
    def place_order_with_approval(self, action, quantity, price):
        """Place order with manual approval"""
        
        logger.warning(f"⚠️  PENDING ORDER APPROVAL")
        logger.warning(f"   Action: {action}")
        logger.warning(f"   Quantity: {quantity}")
        logger.warning(f"   Price: ${price:.2f}")
        
        approval = input("   Approve? (y/n): ").strip().lower()
        
        if approval == 'y':
            if action == "BUY":
                return self.place_buy_order(self.symbol, quantity)
            else:
                return self.place_sell_order(self.symbol, quantity)
        else:
            logger.info("   ❌ Order REJECTED")
            return None
    
    def run_live_session(self, days=30, require_approval=True):
        """Run live trading session
        
        Parameters
        ----------
        days : int
            Number of days of historical data to use
        require_approval : bool
            If True, require manual approval for each order
        """
        
        if not self.connect():
            logger.error("Cannot proceed without IB connection")
            return
        
        try:
            # Load historical data
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            logger.info(f"\n📊 Loading {self.symbol} data from {start_date} to {end_date}")
            prices, volumes = self.env.load_data(start_date, end_date)
            
            if len(prices) == 0:
                logger.error("No data loaded")
                return
            
            logger.info(f"✅ Loaded {len(prices)} bars")
            logger.info(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
            
            # Reset environment
            state = self.env.reset()
            logger.info(f"\n🎯 Starting live trading session")
            logger.info(f"   Initial balance: ${self.env.initial_balance:.2f}")
            logger.info(f"   Current price: ${prices[self.env.current_step]:.2f}")
            logger.info(f"   Manual approval: {'ENABLED' if require_approval else 'DISABLED'}")
            
            # Run trading loop
            step_count = 0
            trades_executed = []
            max_steps = len(prices) - 1
            
            while step_count < max_steps:
                current_price = prices[self.env.current_step]
                
                # Simple strategy: Buy at VAL, Sell at VAH
                if self.env.shares_held == 0 and current_price <= self.env.val:
                    # BUY signal
                    max_shares = int(self.env.balance / current_price)
                    if max_shares > 0:
                        quantity = max(1, max_shares // 2)
                        logger.info(f"\n💰 BUY Signal at ${current_price:.2f} (VAL: ${self.env.val:.2f})")
                        
                        # Place live order
                        if require_approval:
                            trade = self.place_order_with_approval("BUY", quantity, current_price)
                        else:
                            trade = self.place_buy_order(self.symbol, quantity)
                        
                        if trade:
                            trades_executed.append(("BUY", quantity, current_price))
                
                elif self.env.shares_held > 0 and current_price >= self.env.vah:
                    # SELL signal
                    logger.info(f"\n💵 SELL Signal at ${current_price:.2f} (VAH: ${self.env.vah:.2f})")
                    
                    # Place live order
                    if require_approval:
                        trade = self.place_order_with_approval("SELL", self.env.shares_held, current_price)
                    else:
                        trade = self.place_sell_order(self.symbol, self.env.shares_held)
                    
                    if trade:
                        trades_executed.append(("SELL", self.env.shares_held, current_price))
                
                # Execute environment step
                state, reward, done = self.env.step(0)  # HOLD action
                step_count += 1
                
                if step_count % 5 == 0:
                    logger.info(f"Step {step_count}/{max_steps}: Price=${current_price:.2f}, "
                              f"Shares={self.env.shares_held}, "
                              f"Balance=${self.env.balance:.2f}")
                
                # Continue processing all bars regardless of done flag
                # if done:
                #     logger.info("Episode done!")
                #     break
            
            # Summary
            logger.info("\n" + "="*70)
            logger.info("📊 TRADING SESSION SUMMARY")
            logger.info("="*70)
            logger.info(f"Total steps: {step_count}")
            logger.info(f"Trades executed: {len(trades_executed)}")
            for i, (action, qty, price) in enumerate(trades_executed, 1):
                logger.info(f"  {i}. {action}: {qty} x {self.symbol} @ ${price:.2f}")
            
            final_price = prices[-1] if len(prices) > 0 else 0
            final_value = self.env.balance + self.env.shares_held * final_price
            
            logger.info(f"\nFinal portfolio:")
            logger.info(f"  Balance: ${self.env.balance:.2f}")
            logger.info(f"  Shares held: {self.env.shares_held}")
            logger.info(f"  Current price: ${final_price:.2f}")
            logger.info(f"  Total value: ${final_value:.2f}")
            logger.info(f"  P&L: ${final_value - self.env.initial_balance:.2f}")
            
        finally:
            self.disconnect()


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("🚀 LIVE TRADING SYSTEM - IBKR Integration")
    logger.info("="*70 + "\n")
    
    # ⚠️  IMPORTANT: Ensure IB Gateway is running in PAPER TRADING MODE!
    
    # Create live agent
    agent = LiveTradingAgent(symbol="AAPL", initial_balance=10000)
    
    # Run live session (30 days) with manual approval enabled for safety
    agent.run_live_session(days=30, require_approval=True)
