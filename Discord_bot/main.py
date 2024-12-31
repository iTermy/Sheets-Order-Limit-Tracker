import discord
from discord.ext import tasks, commands
import datetime
from googleapiclient.discovery import build
import MetaTrader5 as mt5
import pytz
import json

# Custom Inputs
SPREADSHEET_ID = ''
API_KEY = ''
DISCORD_TOKEN = ''
CHANNEL_ID = ''

# Sheet ranges
RANGE_NAMES = [
    'Daily Trades!B3:K100',
    'Scalps!B3:K59',
    'FX Exotics!B3:K59',
    'Gold!B3:K59',
    'Price Action!B3:K59',
    'Oil!B3:K59',
    'Indices!B3:K59',
    'Crypto!B3:K59',
    'Stocks!B3:K59',
    'Swing!B3:K59',
    'OT!B3:K59',
    'JR!B3:K59'
]

# Default alert distances configuration
DEFAULT_ALERTS = {
    'forex': 10,
    'gold': 10,
    'silver': 0.1,
    'oil': 0.2,
    'nikkei': 100,
    'us30': 100,
    'spx': 10,
    'nas': 50,
    'dax': 10,
    'btc': 300,
    'eth': 10,
    'stocks': 5
}

# Symbol mappings
SYMBOL_TYPES = {
    'gold': 'XAUUSD',
    'silver': 'XAGUSD',
    'oil': 'XTIUSD',
    'nikkei': 'JP225',
    'us30': 'US30',
    'spx': 'US500',
    'nas': 'USTEC',
    'dax': 'DE40',
    'btc': 'BTCUSD',
    'eth': 'ETHUSD'
}


def get_forex_prices(symbol: str) -> list or None:
    """
    Get the current bid and ask prices for a symbol.

    Returns:
    [symbol, bid, ask]
    """
    if not mt5.initialize():
        print(f"MT5 failed to initialize, error code = {mt5.last_error()}")
        return None

    try:
        symbol_info_tick = mt5.symbol_info_tick(symbol)
        if symbol_info_tick is None:
            print(f"Failed to get symbol info for {symbol}")
            return None
        return [symbol, symbol_info_tick.bid, symbol_info_tick.ask]
    except Exception as e:
        print(f"Error getting prices: {e}")
        return None


def is_forex_pair(symbol: str) -> bool:
    """Determine if the symbol is a forex pair."""
    # List of all currencies (I think)
    currency_codes = {'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'SGD', 'HKD'}

    if len(symbol) == 6:
        first_currency = symbol[:3]
        second_currency = symbol[3:]
        return first_currency in currency_codes and second_currency in currency_codes
    return False


def calculate_active_limit_distances(order_limits):
    """
    Calculate distances for active limits.

    Args:
    order_limits (List): List of lists containing each group of limits.
        Example:
            [[symbol, position, sl, price 1, price 2, price 3, price 4, price 5, price 6, status], [symbol...]]

    Returns:
        active_limits_with_distances (List): List of lists containing active limits and their distances to current price
        Example:
            [[symbol, position, sl, price 1, price 2, status, distance], [symbol, position, sl, price 1, price 2...]]
    """
    active_limits_with_distances = []
    unique_symbols = set()

    for group in order_limits:
        status = group[-1]
        if status.lower() not in ["cancelled", "nm", "near miss", "expired"]:
            unique_symbols.add(group[0])

    symbol_prices = {}
    for symbol in unique_symbols:
        forex_price = get_forex_prices(symbol)
        if forex_price is not None:
            symbol_prices[symbol] = forex_price[1]

    for group in order_limits:
        symbol = group[0]
        limit_price = float(group[3])
        status = group[-1]

        if status.lower() in ["cancelled", "nm", "near miss", "expired"]:
            continue

        try:
            if symbol in symbol_prices:
                current_bid_price = symbol_prices[symbol]

                if is_forex_pair(symbol):
                    pip_size = 0.01 if "JPY" in symbol else 0.0001
                    digits = 2 if "JPY" in symbol else 5
                    distance = round(abs(current_bid_price - limit_price) / pip_size, digits)
                else:
                    distance = round(abs(current_bid_price - limit_price), 2)

                active_limits_with_distances.append(group + [distance])
            else:
                active_limits_with_distances.append(group + ["ERROR"])

        except Exception as e:
            active_limits_with_distances.append(group + ["ERROR"])

    return active_limits_with_distances


class PriceAlertBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='$', intents=discord.Intents.all())
        self.remove_command('help')
        self.sheet_data_cache = []
        self.price_alerts_cache = {}
        self.alert_channel = None
        self.alert_distances = self.load_alert_distances()

        if not mt5.initialize():
            print(f"MT5 initialization failed")
            return

    def load_alert_distances(self):
        try:
            with open('alert_distances.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # If file doesn't exist, use defaults
            self.save_alert_distances(DEFAULT_ALERTS)
            return DEFAULT_ALERTS

    def save_alert_distances(self, distances):
        with open('alert_distances.json', 'w') as f:
            json.dump(distances, f, indent=4)

    def get_symbol_type(self, symbol):
        if symbol.endswith(('.NYSE', '.NAS')):
            return 'stocks'

        for sym_type, sym in SYMBOL_TYPES.items():
            if symbol == sym:
                return sym_type

        return 'forex'

    async def setup_hook(self):
        self.check_prices.start()
        self.update_sheet_data.start()
        await self.add_cog(PriceAlertCommands(self))

    async def on_ready(self):
        print(f'Logged in as {self.user}')
        self.alert_channel = self.get_channel(int(CHANNEL_ID))
        if not self.alert_channel:
            print(f"Couldn't find channel with ID {CHANNEL_ID}")

    def clean_old_alerts(self):
        current_time = datetime.datetime.now(pytz.UTC)
        self.price_alerts_cache = {
            k: v for k, v in self.price_alerts_cache.items()
            if current_time - v < datetime.timedelta(hours=24)
        }

    @tasks.loop(seconds=30)
    async def update_sheet_data(self):
        try:
            sheets = build('sheets', 'v4', developerKey=API_KEY).spreadsheets()
            result = sheets.values().batchGet(
                spreadsheetId=SPREADSHEET_ID,
                ranges=RANGE_NAMES
            ).execute()

            all_values = []
            for value_range in result.get('valueRanges', []):
                values = value_range.get('values', [])
                all_values.extend(values)

            self.sheet_data_cache = all_values
            self.clean_old_alerts()

        except Exception as e:
            print(f"Error updating sheet data: {e}")

    @tasks.loop(seconds=5)
    async def check_prices(self):
        if not self.alert_channel:
            return

        try:
            active_limits = calculate_active_limit_distances(self.sheet_data_cache)
            current_time = datetime.datetime.now(pytz.UTC)

            for limit in active_limits:
                if not isinstance(limit[-1], str):
                    symbol = limit[0]
                    symbol_type = self.get_symbol_type(symbol)
                    alert_distance = self.alert_distances[symbol_type]

                    position = limit[1]
                    stop_loss = limit[2]
                    first_limit = limit[3]
                    distance = limit[-1]

                    limit_prices = limit[3:-1]
                    limit_id = f"{symbol}_{first_limit}"

                    if distance <= alert_distance and limit_id not in self.price_alerts_cache:
                        alert_msg = (
                            f"🚨 **Price Alert!**\n"
                            f"Symbol: {symbol}\n"
                            f"Position: {position}\n"
                            f"Stop Loss: {stop_loss}\n"
                            f"Limits:"
                        )

                        for idx, lim_price in enumerate(limit_prices, 1):
                            marker = "•"
                            alert_msg += f"\n{marker} Limit {idx}: {lim_price}"

                        alert_msg += f"\n\nFirst limit is {distance} {'pips' if is_forex_pair(symbol) else 'dollars'} away"

                        await self.alert_channel.send(alert_msg)
                        self.price_alerts_cache[limit_id] = current_time

        except Exception as e:
            print(f"Error checking prices: {e}")

    def cog_unload(self):
        self.check_prices.cancel()
        self.update_sheet_data.cancel()
        mt5.shutdown()


class PriceAlertCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name='alert')
    async def set_alert_distance(self, ctx, symbol_type: str, distance: float):
        symbol_type = symbol_type.lower()

        if symbol_type not in DEFAULT_ALERTS:
            valid_types = ', '.join(DEFAULT_ALERTS.keys())
            await ctx.send(f"Invalid symbol type. Valid types are: {valid_types}")
            return

        if distance <= 0:
            await ctx.send("Distance must be greater than 0")
            return

        self.bot.alert_distances[symbol_type] = distance
        self.bot.save_alert_distances(self.bot.alert_distances)

        unit = "pips" if symbol_type == "forex" else "dollars"
        await ctx.send(f"Alert distance for {symbol_type} set to {distance} {unit}")

    @commands.command(name='config')
    async def show_config(self, ctx):
        config_msg = "**Current Alert Configurations:**\n"

        for symbol_type, distance in self.bot.alert_distances.items():
            unit = "pips" if symbol_type == "forex" else "dollars"
            if symbol_type in SYMBOL_TYPES:
                config_msg += f"{symbol_type.capitalize()} ({SYMBOL_TYPES[symbol_type]}): {distance} {unit}\n"
            elif symbol_type == "stocks":
                config_msg += f"Stocks (.NYSE/.NAS): {distance} {unit}\n"
            else:
                config_msg += f"{symbol_type.capitalize()}: {distance} {unit}\n"

        await ctx.send(config_msg)

    @commands.command(name='help')
    async def help_command(self, ctx):
        help_text = """
        **Available Commands:**
    `$help` - Shows this help message
    `$prefix <new_prefix>` - Changes the bot's command prefix
    `$closest` - Shows the 10 closest active limits
    `$alert <symbol_type> <distance>` - Set alert distance for a symbol type
    `$config` - Show current alert configurations

    **About:**
    This bot monitors trading limits and alerts when prices come within configured distances of your limits.
        """
        await ctx.send(help_text)

    @commands.command(name='prefix')
    async def change_prefix(self, ctx, new_prefix: str):
        self.bot.command_prefix = new_prefix
        await ctx.send(f"Command prefix changed to: {new_prefix}")

    @commands.command(name='closest')
    async def show_closest(self, ctx):
        try:
            active_limits = calculate_active_limit_distances(self.bot.sheet_data_cache)

            valid_limits = [limit for limit in active_limits if not isinstance(limit[-1], str)]
            sorted_limits = sorted(valid_limits, key=lambda x: x[-1])[:10]

            if not sorted_limits:
                await ctx.send("No active limits found.")
                return

            response = "**10 Closest Limits:**\n"
            for limit in sorted_limits:
                symbol = limit[0]
                limit_price = limit[3]
                distance = limit[-1]
                response += f"{symbol} at {limit_price} - {distance} {'pips' if is_forex_pair(symbol) else 'dollars'} away\n"

            await ctx.send(response)

        except Exception as e:
            await ctx.send(f"Error getting closest limits: {e}")


def main():
    bot = PriceAlertBot()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
