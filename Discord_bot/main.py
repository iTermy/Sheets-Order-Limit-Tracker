import discord
from discord.ext import tasks, commands
from discord.ext.commands import BucketType
import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account
import MetaTrader5 as mt5
import pytz
import json
import re
import os
import sys
from typing import Dict, List, Tuple
from googleapiclient.errors import HttpError

# Load credentials
try:
    with open('config.json', 'r') as f:
        config = json.load(f)

    SPREADSHEET_ID = str(config.get('spreadsheet_id', ''))
    API_KEY = str(config.get('api_key', ''))
    DISCORD_TOKEN = str(config.get('discord_token', ''))
    CHANNEL_ID = str(config.get('channel_id', ''))

    if not all([SPREADSHEET_ID, API_KEY, DISCORD_TOKEN, CHANNEL_ID]):
        print("Warning: One or more required configuration values are empty in config.json")

except FileNotFoundError:
    print("Error: config.json not found. Please create a config.json file with required credentials.")
    sys.exit(1)
except json.JSONDecodeError:
    print("Error: config.json is not valid JSON. Please check the format.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading config.json: {str(e)}")
    sys.exit(1)

# Sheet ranges
RANGE_NAMES = [
    'Daily Trades!B3:K100',
    'Scalps!B3:K59',
    'Exotics!B3:K59',
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
    'gold': 5,
    'silver': 0.1,
    'oil': 0.2,
    'nikkei': 100,
    'us30': 100,
    'spx': 10,
    'nas': 50,
    'dax': 10,
    'btc': 750,
    'eth': 20,
    'stocks': 3
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

# I have no idea why I can't combine this with the top one. It breaks for some reason.
SYMBOL_MAPPINGS = {
    'gold': 'XAUUSD',
    'dax': 'DE40',
    'spx': 'US500',
    'nas': 'USTEC',
    'btc': 'BTCUSD',
    'eth': 'ETHUSD',
    'gu': 'GBPUSD',
    'uj': 'USDJPY'
}

SHEET_NAME_MAPPING = {
    'daily trades': 'Daily Trades',
    'scalps': 'Scalps',
    'exotics': 'Exotics',
    'gold': 'Gold',
    'price action': 'Price Action',
    'oil': 'Oil',
    'indices': 'Indices',
    'crypto': 'Crypto',
    'stocks': 'Stocks',
    'swing': 'Swing',
    'ot': 'OT',
    'jr': 'JR'
}

FOREX_MAJORS = {
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
    'AUDUSD', 'USDCAD', 'NZDUSD'
}

PAGINATION_EMOJI = ['⬅️', '➡️']


def add_limit_to_sheet(limit_data: list) -> bool:
    """
    Add a limit entry to the specified Google Sheet.

    Args:
        limit_data (list): [Sheet, Date, Symbol, Position, SL, Limit1-6, Status, Comments]

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Verify data format
        if len(limit_data) != 13:  # Sheet + 12 columns (A through L)
            raise ValueError("Invalid data format. Expected 13 items including sheet name.")

        sheet_name = limit_data[0]
        row_data = limit_data[1:]  # Remove sheet name from data

        # Initialize Google Sheets API with service account
        try:
            credentials = service_account.Credentials.from_service_account_file(
                'keys.json',
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            service = build('sheets', 'v4', credentials=credentials)
        except Exception as e:
            raise InvalidCredentials(f"Failed to authenticate with Google Sheets API: {str(e)}")

        # Find the correct sheet range
        valid_sheets = [name.split('!')[0] for name in RANGE_NAMES]
        if sheet_name not in valid_sheets:
            raise ValueError(f"Invalid sheet name. Valid sheets are: {', '.join(valid_sheets)}")

        # Get current values to find empty row
        sheet_range = f"{sheet_name}!A3:L100"
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=sheet_range
        ).execute()

        values = result.get('values', [])

        # Find first empty row
        empty_row = None
        for i in range(98):
            row_index = i + 3

            if i >= len(values):
                empty_row = row_index
                break

            row = values[i] if i < len(values) else []
            if not row or all(cell == '' for cell in row):
                empty_row = row_index
                break

        if empty_row is None:
            raise ValueError("No empty rows found in range (3-100)")

        # Prepare update range and values
        update_range = f"{sheet_name}!A{empty_row}:L{empty_row}"

        # Update the sheet
        body = {
            'values': [row_data]
        }

        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=update_range,
            valueInputOption='RAW',
            body=body
        ).execute()

        return True

    except HttpError as e:
        if e.resp.status in [401, 403]:
            raise InvalidCredentials(f"Invalid or expired credentials: {str(e)}")
        raise
    except Exception as e:
        print(f"Error adding limit to sheet: {str(e)}")
        return False


async def update_limit_status(symbol: str, price1: str, new_status: str) -> list:
    """
    Update the status of limits matching symbol and first limit price.

    Args:
        symbol (str): Trading symbol
        price1 (str): First limit price
        new_status (str): New status to set

    Returns:
        list: List of sheets where updates were made
    """
    try:
        # Initialize Google Sheets API
        credentials = service_account.Credentials.from_service_account_file(
            'keys.json',
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        service = build('sheets', 'v4', credentials=credentials)

        updated_sheets = []
        checked_sheets = set()

        # First check most likely sheet based on symbol
        if symbol == 'XAUUSD':
            priority_sheet = 'Gold!B3:K100'
        elif symbol in FOREX_MAJORS:
            priority_sheet = 'Daily Trades!B3:K100'
        elif symbol in ['US500', 'USTEC', 'DE40']:
            priority_sheet = 'Indices!B3:K100'
        elif symbol in ['BTCUSD', 'ETHUSD']:
            priority_sheet = 'Crypto!B3:K100'
        else:
            priority_sheet = None

        if priority_sheet:
            result = await check_and_update_sheet(service, priority_sheet, symbol, price1, new_status)
            if result:
                updated_sheets.append(priority_sheet.split('!')[0])
            checked_sheets.add(priority_sheet)

        # If not found in priority sheet, check all other sheets
        if not updated_sheets:
            for range_name in RANGE_NAMES:
                if range_name not in checked_sheets:
                    result = await check_and_update_sheet(service, range_name, symbol, price1, new_status)
                    if result:
                        updated_sheets.append(range_name.split('!')[0])

        return updated_sheets

    except Exception as e:
        print(f"Error updating status: {str(e)}")
        return []


async def check_and_update_sheet(service, range_name: str, symbol: str, price1: str, new_status: str) -> bool:
    """Helper function to check and update a specific sheet."""
    try:
        # Get current values
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name
        ).execute()

        values = result.get('values', [])
        updates_needed = []

        # Check each row
        for row_idx, row in enumerate(values):
            if len(row) >= 4:  # Make sure row has enough columns
                row_symbol = row[0]
                row_price1 = row[3]

                if row_symbol == symbol and str(row_price1) == str(price1):
                    # Calculate the status column position (11th column, index 10)
                    update_range = f"{range_name.split('!')[0]}!K{row_idx + 3}"
                    updates_needed.append({
                        'range': update_range,
                        'values': [[new_status]]
                    })

        # Perform all updates if any matches found
        if updates_needed:
            body = {
                'valueInputOption': 'RAW',
                'data': updates_needed
            }
            service.spreadsheets().values().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body=body
            ).execute()
            return True

        return False

    except Exception as e:
        print(f"Error checking sheet {range_name}: {str(e)}")
        return False


def determine_sheet_name(symbol: str, limit_string: str) -> str:
    """
    Determine the appropriate sheet name based on symbol and limit string.

    Args:
        symbol (str): The trading symbol
        limit_string (str): The full command string

    Returns:
        str: Sheet name
    """
    # Check for explicit sheet name first
    sheet_pattern = '|'.join([
        r'\b' + re.escape(name) + r'\b'
        for name in SHEET_NAME_MAPPING.keys()
    ])

    sheet_match = re.search(sheet_pattern, limit_string.lower())
    if sheet_match:
        matched_name = sheet_match.group(0)
        proper_name = SHEET_NAME_MAPPING[matched_name]
        print(f"Found explicit sheet name: '{matched_name}' -> '{proper_name}'")
        return proper_name

    # Check for "swing" keyword
    if 'swing' in limit_string.lower():
        print("Found 'swing' keyword, using Swing sheet")
        return 'Swing'

    # Check symbol type
    symbol = symbol.upper()
    print(f"Determining sheet based on symbol: {symbol}")

    if symbol == 'XAUUSD':
        return 'Gold'
    elif symbol in ['US500', 'USTEC', 'DE40', 'US30']:
        return 'Indices'
    elif symbol in ['BTCUSD', 'ETHUSD']:
        return 'Crypto'
    elif symbol.endswith(('.NYSE', '.NAS')):
        return 'Stocks'
    elif is_forex_pair(symbol):
        if symbol in FOREX_MAJORS:
            print(f"{symbol} is a major forex pair, using Daily Trades")
            return 'Daily Trades'
        print(f"{symbol} is an exotic forex pair, using Exotics")
        return 'Exotics'

    print(f"No specific sheet rule matched for {symbol}, defaulting to Daily Trades")
    return 'Daily Trades'


def get_mapped_symbol(text: str, available_symbols: set) -> str or None:
    """
    Get the correct symbol from text using mappings and available symbols.

    Args:
        text (str): The input text to search for symbol
        available_symbols (set): Set of valid MT5 symbols

    Returns:
        str: Found symbol or None
    """
    text = text.lower()

    # First check for exact stock symbols (ending in .NYSE or .NAS)
    words = text.upper().split()
    for word in words:
        if word.endswith(('.NYSE', '.NAS')) and word in available_symbols:
            return word

    # Then check symbol mappings
    for key, mapped_symbol in SYMBOL_TYPES.items():
        if key in text:
            return mapped_symbol if mapped_symbol in available_symbols else None

    # If no mapping found, look for direct symbol match
    for word in words:
        if word in available_symbols:
            return word

    # Finally, check company names and symbols in each word
    skip_words = {'long', 'short', 'vth', 'hot', 'stops', 'comments'}
    words = [word.lower() for word in text.split() if word.lower() not in skip_words]
    words = [word for word in words if not any(c.isdigit() for c in word)]

    for word in words:
        matches = []
        for symbol in available_symbols:
            if symbol.endswith(('.NYSE', '.NAS')):
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info and symbol_info.description:
                    # Check both symbol and description
                    if (word in symbol_info.description.lower() or
                            word in symbol.lower()):
                        matches.append(symbol)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            continue

    return None


def get_bid_price(symbol: str) -> float or None:
    if not mt5.initialize():
        print(f"MT5 failed to initialize, error code = {mt5.last_error()}")
        return None

    try:
        symbol_info_tick = mt5.symbol_info_tick(symbol)
        if symbol_info_tick is None:
            print(f"Failed to get symbol info for {symbol}")
            return None
        return symbol_info_tick.bid
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
        return None


def is_forex_pair(symbol: str) -> bool:
    """Determine if the symbol is a forex pair."""
    currency_codes = {'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'SGD', 'HKD'}

    if len(symbol) == 6:
        first_currency = symbol[:3]
        second_currency = symbol[3:]
        return first_currency in currency_codes and second_currency in currency_codes
    return False


def calculate_active_limit_distances(order_limits: list[list[str]]) -> list[list[str]]:
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
    symbol_prices = {}
    status_exempt = ["cancelled", "nm", "near miss", "expired", "hit", "cancel", "tp", "sl", "stop loss"]

    for group in order_limits:
        status = group[-1].lower()

        # Skip if status is exempt
        if status in status_exempt:
            continue

        symbol = group[0]
        try:
            # Get current price
            if symbol not in symbol_prices:
                current_bid_price = get_bid_price(symbol)
                if current_bid_price is None:
                    active_limits_with_distances.append(group + ["ERROR"])
                    continue
                symbol_prices[symbol] = current_bid_price

            current_bid_price = symbol_prices[symbol]
            limit_price = float(group[3])  # First limit price

            # Calculate distance based on symbol type
            if is_forex_pair(symbol):
                pip_size = 0.01 if "JPY" in symbol else 0.0001
                digits = 2 if "JPY" in symbol else 5
                distance = round(abs(current_bid_price - limit_price) / pip_size, digits)
            else:
                distance = round(abs(current_bid_price - limit_price), 2)

            active_limits_with_distances.append(group + [distance])

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            active_limits_with_distances.append(group + ["ERROR"])

    return active_limits_with_distances


def get_stock_info(symbol: str) -> Tuple[str, str]:
    """
    Get stock symbol and name from MT5.

    Args:
        symbol: The stock symbol

    Returns:
        Tuple containing (symbol, description or symbol)
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info and symbol_info.description:
            return symbol, symbol_info.description
        return symbol, symbol
    except Exception as e:
        print(f"Error getting stock info for {symbol}: {e}")
        return symbol, symbol


def filter_and_sort_stocks(available_symbols: set, start_letter: str = None) -> List[Tuple[str, str]]:
    """
    Filter and sort stocks from available symbols.

    Args:
        available_symbols: Set of available MT5 symbols
        start_letter: Optional letter to filter by

    Returns:
        List of tuples containing (symbol, company_name)
    """
    # Filter for NYSE and NASDAQ stocks
    stocks = [
        symbol for symbol in available_symbols
        if symbol.endswith(('.NYSE', '.NAS'))
    ]

    # Filter by starting letter if provided
    if start_letter:
        stocks = [
            symbol for symbol in stocks
            if symbol.startswith(start_letter.upper())
        ]

    # Get stock info and sort
    stock_info = [get_stock_info(symbol) for symbol in stocks]
    return sorted(stock_info, key=lambda x: x[0])


class InvalidCredentials(Exception):
    pass


class LastOperation:
    def __init__(self, operation_type: str, sheet_name: str, row_index: int,
                 old_values: list, new_values: list, range_name: str):
        self.operation_type = operation_type  # handles 'add_limit', 'status_update', 'expiredaily(weekly)'
        self.sheet_name = sheet_name
        self.row_index = row_index
        self.old_values = old_values
        self.new_values = new_values
        self.range_name = range_name
        self.timestamp = datetime.datetime.now(pytz.UTC)


class PriceAlertBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='$$', intents=discord.Intents.all())
        self.remove_command('help')
        self.sheet_data_cache = []
        self.price_alerts_cache = {}
        self.alert_channel = None
        self.alert_distances = self.load_alert_distances()
        self.alert_cooldown = 24
        self.last_operation = None

        if not mt5.initialize():
            print(f"MT5 initialization failed")
            return

        # Get all available symbols
        symbols = mt5.symbols_get()
        self.available_symbols = {symbol.name for symbol in symbols} if symbols else set()

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
            if current_time - v < datetime.timedelta(hours=self.alert_cooldown)
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
                            f"Limits:"
                        )

                        for idx, lim_price in enumerate(limit_prices, 1):
                            marker = "•"
                            alert_msg += f"\n{marker} Limit {idx}: {lim_price}"

                        alert_msg += f"\nStop Loss: {stop_loss}\n"
                        alert_msg += f"\nFirst limit is {distance} {'pips' if is_forex_pair(symbol) else 'dollars'} away"

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
        self.stock_pages: Dict[int, List[str]] = {}

    @commands.command(name='alert')
    async def set_alert_distance(self, ctx, symbol_type: str, distance: float):
        """Changes how close a limit should be before alerting."""
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
        """Shows the current alert configurations. Determines how close a limit should be before alerting."""
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
        help_text = f"""
        **Available Commands:**
    `{self.bot.command_prefix}help` - Shows this help message
    `{self.bot.command_prefix}prefix <new_prefix>` - Changes the bot's command prefix
    `{self.bot.command_prefix}closest` - Shows the 10 closest active limits
    `{self.bot.command_prefix}alert <symbol_type> <distance>` - Set alert distance for a symbol type
    `{self.bot.command_prefix}config` - Show current alert configurations

    **About:**
    This bot monitors trading limits and alerts when prices come within configured distances of your limits.
        """
        await ctx.send(help_text)

    @commands.command(name='prefix')
    async def change_prefix(self, ctx, new_prefix: str):
        """Change bot prefix"""
        self.bot.command_prefix = new_prefix
        await ctx.send(f"Command prefix changed to: {new_prefix}")

    @commands.command(name='closest')
    @commands.cooldown(1, 2, BucketType.user)
    async def show_closest(self, ctx):
        """Displays the 10 closest valid limits"""
        try:
            # Retrieve distances for each limit
            active_limits = calculate_active_limit_distances(self.bot.sheet_data_cache)

            # Sort the valid limits by closest distance (First 10)
            valid_limits = [limit for limit in active_limits if not isinstance(limit[-1], str)]
            sorted_limits = sorted(valid_limits, key=lambda x: x[-1])[:10]

            if not sorted_limits:
                await ctx.send("No active limits found.")
                return

            # Compose message
            response = "**10 Closest Limits:**\n"
            for limit in sorted_limits:
                symbol = limit[0]
                limit_price = limit[3]
                distance = limit[-1]
                response += f"{symbol} at {limit_price} - {distance} {'pips' if is_forex_pair(symbol) else 'dollars'} away\n"

            await ctx.send(response)

        except Exception as e:
            await ctx.send(f"Error getting closest limits: {e}")

    @commands.command(name='add')
    @commands.cooldown(1, 5, BucketType.user)
    async def add_limit(self, ctx, *, limit_string: str):
        """
        Add a group of order limits to the sheet.

        Args: limit_string (str): A string containing the symbol, position, limits (max 6), stop loss, sheet (opt.),
        comments (opt.) Example: $add 1.00724---1.00561 eurusd long Stops 1.00319 scalps Comments: VTH

        Returns:
            None
        """
        try:
            # Check for keys.json
            if not os.path.exists('keys.json'):
                await ctx.send("To use this feature, you must have a keys.json file. None was found.")
                return

            # Find symbol using mappings and available symbols
            symbol = get_mapped_symbol(limit_string, self.bot.available_symbols)
            if not symbol:
                await ctx.send("Error: No valid trading symbol found in string")
                return

            # Determine sheet name based on symbol and limit string
            sheet_name = determine_sheet_name(symbol, limit_string)

            # Get current date
            current_date = datetime.datetime.now().strftime("%B %d")

            # Find position (long/short)
            position_match = re.search(r'\b(long|short)\b', limit_string.lower())
            if not position_match:
                await ctx.send("Error: Position (long/short) not found in string")
                return
            position = position_match.group(1).upper()

            # Get all numbers
            numbers = re.findall(r'(\d+\.?\d*)', limit_string)
            if not numbers:
                await ctx.send("Error: No numbers found in string")
                return

            # Convert large numbers if needed (Ex: AUDUSD is sometimes written as 61234 instead of 0.61234)
            if float(numbers[1]) > 30000 and symbol not in ["US30", "JP225", "BTCUSD", "USTEC"]:
                numbers = [str(float(num) / 100000) for num in numbers]

            # Last number is stop loss, rest are limits
            stop_loss = numbers[-1]
            limits = numbers[:-1]

            # Fill remaining limits with empty strings (total 6 limits)
            limits.extend([''] * (6 - len(limits)))

            # Get comments and auto-keywords
            comments = ''
            comments_match = re.search(r'Comments:(.*?)(?=$|\n)', limit_string, re.IGNORECASE)
            if comments_match:
                comments = comments_match.group(1).strip()

            # Add auto-keywords
            auto_keywords = []
            if 'hot' in limit_string.lower():
                auto_keywords.append('HOT')
            if 'vth' in limit_string.lower():
                auto_keywords.append('VTH')

            if auto_keywords:
                if comments:
                    comments = f"{comments} {', '.join(auto_keywords)}"
                else:
                    comments = f"{', '.join(auto_keywords)}"

            # Prepare data for sheet
            limit_data = [
                sheet_name,
                current_date,
                symbol,
                position,
                stop_loss,
                *limits,  # All 6 limits (some may be empty strings)
                '',  # Status (always blank)
                comments
            ]

            print(f"Adding limit to {sheet_name} sheet:", limit_data)

            # Add to sheet
            if add_limit_to_sheet(limit_data):
                await ctx.send(f"Successfully added limit to {sheet_name} sheet")
            else:
                await ctx.send("Failed to add limit to sheet")

        except Exception as e:
            await ctx.send(f"Error processing command: {str(e)}")
            print(f"Error in add_limit command: {str(e)}")


    @commands.command(name='all')
    @commands.cooldown(1, 2, BucketType.user)
    async def show_all(self, ctx):
        """Show all active limits, sorted alphabetically by symbol, then by ascending price."""
        try:
            active_limits = calculate_active_limit_distances(self.bot.sheet_data_cache)

            valid_limits = []
            invalid_limits = []

            for limit in active_limits:
                if isinstance(limit[-1], str):  # Invalid limits have string distance ("ERROR")
                    invalid_limits.append(limit)
                else:
                    valid_limits.append(limit)

            # Sort valid limits by symbol then by first limit price
            sorted_limits = sorted(valid_limits, key=lambda x: (x[0], float(x[3])))

            if not sorted_limits and not invalid_limits:
                await ctx.send("No active limits found.")
                return

            response = "**All Active Limits:**\n"

            # Add valid limits
            for limit in sorted_limits:
                symbol = limit[0]
                limit_price = limit[3]
                distance = limit[-1]
                response += f"{symbol} at {limit_price} - {distance} {'pips' if is_forex_pair(symbol) else 'dollars'} away\n"

            # Add invalid limits if any exist
            if invalid_limits:
                response += "\n**Limits with Errors:**\n"
                for limit in invalid_limits:
                    symbol = limit[0]
                    limit_price = limit[3]
                    response += f"{symbol} at {limit_price} - Error getting distance\n"

            # Handling long messages
            if len(response) > 2000:
                response_parts = []
                current_part = ""

                for line in response.split('\n'):
                    if len(current_part) + len(line) + 1 > 2000:
                        response_parts.append(current_part)
                        current_part = line + '\n'
                    else:
                        current_part += line + '\n'

                if current_part:
                    response_parts.append(current_part)

                for part in response_parts:
                    await ctx.send(part)
            else:
                await ctx.send(response)

        except Exception as e:
            await ctx.send(f"Error getting limits: {e}")
            print(f"Most likely error in calculate_active_limit_distances. Error: {str(e)}")

    @commands.command(name='status')
    @commands.cooldown(1, 5, BucketType.user)
    async def update_status(self, ctx, *args):
        """Update the status of a limit identified by symbol and first limit price."""
        try:
            # Check if we have exactly 3 arguments
            if len(args) != 3:
                await ctx.send(
                    "Error: Command requires exactly 3 parts: symbol, 1st limit price, and status\n"
                    "Example: $status EURUSD 1.0500 hit")
                return

            symbol, price1, status = args

            symbol = symbol.upper()

            if not os.path.exists('keys.json'):
                await ctx.send("To use this feature, you must have a keys.json file. None was found.")
                return

            if len(status.split()) > 1:
                await ctx.send("Error: Status must be a single word")
                return

            # Update the status
            updated_sheets = await update_limit_status(symbol, price1, status)

            if updated_sheets:
                await ctx.send(f"Successfully updated status of {symbol} {price1} to {status}")
            else:
                await ctx.send(f"No limits found matching {symbol} with first limit price {price1}")

        except Exception as e:
            await ctx.send(f"Error updating status: {str(e)}")
            print(f"Error in status command: {str(e)}")

    @commands.command(name='stocklist')
    async def stock_list(self, ctx, *, query: str = None):
        """
        Show available stocks, optionally filtered by starting letter or company name.

        Args:
            query: Optional filter - either a single letter for listing or company name to search
        """
        try:
            if not query:
                # Show all stocks if no query provided
                stocks = filter_and_sort_stocks(self.bot.available_symbols)
            elif len(query) == 1 and query.isalpha():
                # Existing behavior for single letter
                stocks = filter_and_sort_stocks(self.bot.available_symbols, query)
            else:
                # Search for specific company
                query = query.lower()
                matches = []

                for symbol in self.bot.available_symbols:
                    if symbol.endswith(('.NYSE', '.NAS')):
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info and symbol_info.description:
                            if query in symbol_info.description.lower() or query in symbol.lower():
                                matches.append((symbol, symbol_info.description))

                if matches:
                    if len(matches) == 1:
                        # Single match found
                        symbol, name = matches[0]
                        await ctx.send(f"Found match: `{symbol}` - {name}")
                        return
                    else:
                        # Multiple matches found
                        stocks = sorted(matches, key=lambda x: x[0])
                else:
                    await ctx.send(f"No stocks found matching '{query}'.")
                    return

            if not stocks:
                filter_msg = f" starting with '{query}'" if len(query) == 1 else ""
                await ctx.send(f"No stocks found{filter_msg}.")
                return

            # Paginate results
            pages = []
            for i in range(0, len(stocks), 20):
                page_stocks = stocks[i:i + 20]
                page = "**Available Stocks:**\n\n"
                for symbol, name in page_stocks:
                    page += f"`{symbol}` - {name}\n"
                page += f"\nPage {len(pages) + 1} of {(len(stocks) + 19) // 20}"
                pages.append(page)

            # Store pages for this message
            message = await ctx.send(pages[0])
            self.stock_pages[message.id] = pages

            # Add navigation reactions
            if len(pages) > 1:
                for emoji in PAGINATION_EMOJI:
                    await message.add_reaction(emoji)

        except Exception as e:
            await ctx.send(f"Error listing stocks: {str(e)}")
            print(f"Error in stock_list command: {str(e)}")

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        """Handle pagination reactions for $stocklist."""
        if user.bot or str(reaction.emoji) not in PAGINATION_EMOJI:
            return

        message = reaction.message
        if message.id not in self.stock_pages:
            return

        pages = self.stock_pages[message.id]
        current_page = int(message.content.split('\n')[-1].split('Page')[-1].split('of')[0].strip())
        total_pages = len(pages)

        # Calculate new page
        if str(reaction.emoji) == '➡️':
            new_page = 0 if current_page == total_pages else current_page
        else:  # Left arrow
            new_page = total_pages - 1 if current_page == 1 else current_page - 2

        try:
            await message.edit(content=pages[new_page])
        except Exception as e:
            print(f"Error updating page: {e}")
            print(f"Current page: {current_page}, New page: {new_page}, Total pages: {total_pages}")

        await reaction.remove(user)

    @commands.Cog.listener()
    async def on_message_delete(self, message):
        """Clean up stored pages when message is deleted."""
        if message.id in self.stock_pages:
            del self.stock_pages[message.id]

    @commands.command(name='setalertcooldown')
    @commands.cooldown(1, 2, BucketType.user)
    async def set_alert_cooldown(self, ctx, hours: float):
        """Set the cooldown period for price alerts."""
        try:
            if hours <= 0:
                await ctx.send("Error: Cooldown must be greater than 0 hours")
                return

            self.bot.alert_cooldown = hours
            await ctx.send(f"Alert cooldown period set to {hours} hours")

        except ValueError:
            await ctx.send("Error: Please provide a valid number of hours")
        except Exception as e:
            await ctx.send(f"Error setting alert cooldown: {str(e)}")

    @commands.command(name='clearalertcooldown')
    @commands.cooldown(1, 2, BucketType.user)
    async def clear_alert_cooldown(self, ctx):
        """Clear all alert cooldowns."""
        try:
            self.bot.price_alerts_cache.clear()
            await ctx.send("All alert cooldowns have been cleared")

        except Exception as e:
            await ctx.send(f"Error clearing alert cooldowns: {str(e)}")

    @commands.command(name='expiredaily')
    @commands.cooldown(1, 10, BucketType.user)
    async def expire_daily(self, ctx):
        """Mark trades in daily-related sheets as expired unless VTH."""
        try:
            if not os.path.exists('keys.json'):
                await ctx.send("To use this feature, you must have a keys.json file. None was found.")
                return

            credentials = service_account.Credentials.from_service_account_file(
                'keys.json',
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            service = build('sheets', 'v4', credentials=credentials)

            daily_sheets = ['Daily Trades', 'Scalps', 'OT', 'Indices']
            total_updates = 0

            for sheet_name in daily_sheets:
                range_name = f'{sheet_name}!B3:L100'

                result = service.spreadsheets().values().get(
                    spreadsheetId=SPREADSHEET_ID,
                    range=range_name
                ).execute()

                values = result.get('values', [])
                updates_needed = []

                for row_idx, row in enumerate(values):
                    # Pad the row with empty strings if it's shorter than 11 columns
                    padded_row = row + [''] * (11 - len(row)) if len(row) < 11 else row
                    status = padded_row[9].strip()
                    comments = padded_row[10].lower()

                    # Update if: status is empty AND comments don't contain 'vth'
                    if not status and 'vth' not in comments:
                        update_range = f'{sheet_name}!K{row_idx + 3}'
                        updates_needed.append({
                            'range': update_range,
                            'values': [['expired']]
                        })

                # Perform batch update if any updates needed
                if updates_needed:
                    body = {
                        'valueInputOption': 'RAW',
                        'data': updates_needed
                    }
                    service.spreadsheets().values().batchUpdate(
                        spreadsheetId=SPREADSHEET_ID,
                        body=body
                    ).execute()
                    total_updates += len(updates_needed)

            await ctx.send(f"Operation complete: {total_updates} trades marked as expired across daily-related sheets.")

        except Exception as e:
            await ctx.send(f"Error updating status: {str(e)}")
            print(f"Error in expire_daily command: {str(e)}")

    @commands.command(name='expireweekly')
    @commands.cooldown(1, 10, BucketType.user)
    async def expire_weekly(self, ctx):
        """Mark trades in all sheets (except Stocks) as expired."""
        try:
            if not os.path.exists('keys.json'):
                await ctx.send("To use this feature, you must have a keys.json file. None was found.")
                return

            credentials = service_account.Credentials.from_service_account_file(
                'keys.json',
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            service = build('sheets', 'v4', credentials=credentials)

            # All sheets except Stocks
            sheets = [name.split('!')[0] for name in RANGE_NAMES if not name.startswith('Stocks!')]
            total_updates = 0

            for sheet_name in sheets:
                range_name = f'{sheet_name}!B3:L100'

                result = service.spreadsheets().values().get(
                    spreadsheetId=SPREADSHEET_ID,
                    range=range_name
                ).execute()

                values = result.get('values', [])
                updates_needed = []

                for row_idx, row in enumerate(values):
                    # Pad the row with empty strings if it's shorter than 11 columns
                    padded_row = row + [''] * (11 - len(row)) if len(row) < 11 else row
                    status = padded_row[9].strip()
                    comments = padded_row[10].lower()

                    # Update if: status is empty AND comments don't contain 'alien' or 'maso'
                    if not status and not any(word in comments for word in ['alien', 'maso']):
                        update_range = f'{sheet_name}!K{row_idx + 3}'
                        updates_needed.append({
                            'range': update_range,
                            'values': [['expired']]
                        })

                # Perform batch update if any updates needed
                if updates_needed:
                    body = {
                        'valueInputOption': 'RAW',
                        'data': updates_needed
                    }
                    service.spreadsheets().values().batchUpdate(
                        spreadsheetId=SPREADSHEET_ID,
                        body=body
                    ).execute()
                    total_updates += len(updates_needed)

            await ctx.send(f"Operation complete: {total_updates} trades marked as expired across all non-stock sheets.")

        except Exception as e:
            await ctx.send(f"Error updating status: {str(e)}")
            print(f"Error in expire_weekly command: {str(e)}")

    # Cooldown error handling
    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"Please wait {round(error.retry_after, 1)} seconds before using this command again.")


def main():
    bot = PriceAlertBot()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
