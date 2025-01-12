import discord
from discord.ext import tasks, commands
import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account
import MetaTrader5 as mt5
import pytz
import json
import re
import os
import sys
from googleapiclient.errors import HttpError

# Load credentials from config.json
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
    'gold': 10,
    'silver': 0.1,
    'oil': 0.2,
    'nikkei': 100,
    'us30': 100,
    'spx': 10,
    'nas': 50,
    'dax': 10,
    'btc': 750,
    'eth': 20,
    'stocks': 2
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
    'eth': 'ETHUSD',
    'gu': 'GBPUSD',
    'uj': 'USDJPY'
}

# Sheet mappings
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
            row_index = i + 3  # Start from row 3

            # If we've reached beyond existing values, this row is empty
            if i >= len(values):
                empty_row = row_index
                break

            # If row exists but is empty or has all empty cells
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

        # First check most likely sheet based on symbol (optimization)
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

    # Check symbol mappings first
    for key, mapped_symbol in SYMBOL_TYPES.items():
        if key in text:
            return mapped_symbol if mapped_symbol in available_symbols else None

    # If no mapping found, look for direct symbol match
    words = text.upper().split()
    for word in words:
        if word in available_symbols:
            return word

    return None


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
    unique_symbols = set()

    for group in order_limits:
        status = group[-1]
        status_exempt = ["cancelled", "nm", "near miss", "expired", "hit", "cancel", "tp", "sl", "stop loss"]
        if status.lower() not in status_exempt:
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

        status_exempt = ["cancelled", "nm", "near miss", "expired", "hit", "cancel", "tp", "sl", "stop loss"]
        if status.lower() in status_exempt:
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


class InvalidCredentials(Exception):
    pass


class PriceAlertBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='$$', intents=discord.Intents.all())
        self.remove_command('help')
        self.sheet_data_cache = []
        self.price_alerts_cache = {}
        self.alert_channel = None
        self.alert_distances = self.load_alert_distances()

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
                            f"Limits:"
                        )

                        for idx, lim_price in enumerate(limit_prices, 1):
                            marker = "•"
                            alert_msg += f"\n{marker} Limit {idx}: {lim_price}"

                        alert_msg += f"\nStop Loss: {stop_loss}\n"
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

    @commands.command(name='all')
    async def show_all(self, ctx):
        """Show all active limits, sorted alphabetically by symbol."""
        try:
            active_limits = calculate_active_limit_distances(self.bot.sheet_data_cache)

            # Separate valid and invalid limits
            valid_limits = []
            invalid_limits = []

            for limit in active_limits:
                if isinstance(limit[-1], str):  # Invalid limits have string distance ("ERROR")
                    invalid_limits.append(limit)
                else:
                    valid_limits.append(limit)

            # Sort valid limits by symbol
            sorted_limits = sorted(valid_limits, key=lambda x: x[0])  # x[0] is the symbol

            if not sorted_limits and not invalid_limits:
                await ctx.send("No active limits found.")
                return

            response = "**All Active Limits:**\n"

            # Add valid limits
            for limit in sorted_limits:
                symbol = limit[0]
                limit_price = limit[3]  # First limit price
                distance = limit[-1]
                response += f"{symbol} at {limit_price} - {distance} {'pips' if is_forex_pair(symbol) else 'dollars'} away\n"

            # Add invalid limits if any exist
            if invalid_limits:
                response += "\n**Limits with Errors:**\n"
                for limit in invalid_limits:
                    symbol = limit[0]
                    limit_price = limit[3]
                    response += f"{symbol} at {limit_price} - Error getting distance\n"

            # Split response if it exceeds Discord's message length limit
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
            print(f"Error in show_all command: {str(e)}")


    @commands.command(name='add')
    async def add_limit(self, ctx, *, limit_string: str):
        """Add a new trading limit to the sheet."""
        try:
            # Check for keys.json first
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

            # Convert large numbers if needed (handle indices and crypto)
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
                sheet_name,  # Sheet name
                current_date,  # Date
                symbol,  # Symbol
                position,  # Long/Short
                stop_loss,  # Stop Loss
                *limits,  # All 6 limits (some may be empty strings)
                '',  # Status (always blank)
                comments  # Comments
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

    @commands.command(name='status')
    async def update_status(self, ctx, symbol: str, price1: str, status: str):
        """Update the status of a limit identified by symbol and stop loss."""
        try:
            if not os.path.exists('keys.json'):
                await ctx.send("To use this feature, you must have a keys.json file. None was found.")
                return

            # Validate status is one word
            if len(status.split()) > 1:
                await ctx.send("Error: Status must be a single word")
                return

            # Update the status
            updated_sheets = await update_limit_status(symbol, price1, status)

            if updated_sheets:
                sheets_str = ", ".join(updated_sheets)
                await ctx.send(f"Successfully updated status to '{status}' in sheet(s): {sheets_str}")
            else:
                await ctx.send(f"No limits found matching {symbol} with stop loss {price1}")

        except Exception as e:
            await ctx.send(f"Error updating status: {str(e)}")
            print(f"Error in status command: {str(e)}")


def main():
    bot = PriceAlertBot()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
