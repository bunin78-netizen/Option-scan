import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import time
import os
from dataclasses import dataclass, field


@dataclass
class OptionFilters:
    """Параметры фильтрации"""
    currency: str = "BTC"  # BTC или ETH
    min_volume: float = 1.0  # BTC
    min_open_interest: float = 10.0  # BTC
    iv_min: float = 0.0
    iv_max: float = 2.0  # 200%
    delta_min: float = -1.0
    delta_max: float = 1.0
    dte_min: int = 0
    dte_max: int = 365
    min_liquidity_score: float = 0.0  # (volume * oi) / strike
    exclude_perpetual: bool = True
    instrument_type: str = "all"  # all / inverse / non_inverse
    underlying_price_range: Tuple[float, float] = field(
        default_factory=lambda: (0, float('inf'))
    )  # фильтр по цене базового актива


class DeribitOptionsScanner:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.base_url = "https://www.deribit.com/api/v2"
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()

        # Для хранения исторических данных IV (для расчета percentile)
        self.iv_history: Dict[str, List[float]] = {}

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Универсальный GET запрос с обработкой ошибок"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('error'):
                print(f"API Error: {data['error']['message']}")
                return {}
            return data.get('result', {})
        except Exception as e:
            print(f"Request error: {e}")
            return {}

    def get_instruments(self, currency: str = "BTC", expired: bool = False) -> List[dict]:
        """Получить список всех опционов"""
        params = {
            'currency': currency,
            'kind': 'option',
            'expired': expired
        }
        result = self._get("/public/get_instruments", params)
        return result if isinstance(result, list) else []

    def get_supported_option_currencies(self) -> List[str]:
        """Список валют, для которых биржа отдает option-инструменты."""
        result = self._get("/public/get_currencies")
        if isinstance(result, list):
            currencies = sorted({row.get("currency", "") for row in result if row.get("kind") == "option"})
            return [c for c in currencies if c]
        return ["BTC", "ETH"]

    @staticmethod
    def _parse_instrument_market(name: str) -> dict:
        """Разобрать инструмент на тип рынка (инверсный/неинверсный) и котировку."""
        market_token = name.split("-")[0] if name else ""
        if "_" in market_token:
            base, quote = market_token.split("_", 1)
            return {
                "base_currency": base,
                "quote_currency": quote,
                "pair_type": "non_inverse",
            }

        return {
            "base_currency": market_token,
            "quote_currency": "USD",
            "pair_type": "inverse",
        }

    def get_ticker_batch(self, instruments: List[str]) -> Dict[str, dict]:
        """Батчевое получение данных по тикерам (эффективнее одиночных запросов)"""
        if not instruments:
            return {}

        # Ограничение: 50 инструментов за запрос (rate limit)
        batch_size = 50
        all_data: Dict[str, dict] = {}

        for i in range(0, len(instruments), batch_size):
            batch = instruments[i:i + batch_size]
            params = {'instruments': ','.join(batch)}
            result = self._get("/public/ticker_batch", params)
            if isinstance(result, dict):
                all_data.update(result)
            time.sleep(0.1)  # Rate limit protection (100 req/sec для публичных)

        return all_data

    def calculate_dte(self, expiration_ts: int) -> int:
        """Дней до экспирации"""
        exp_date = datetime.fromtimestamp(expiration_ts / 1000)
        return (exp_date - datetime.now()).days

    def calculate_metrics(self, row: dict, underlying_price: float) -> dict:
        """Расчет дополнительных метрик"""
        metrics: dict = {}

        # Основные данные
        volume = row.get('volume', 0)
        oi = row.get('open_interest', 0)
        strike = row.get('strike', 0)
        iv = row.get('iv', 0)

        # Греки (у Deribit они есть в тикере)
        metrics['delta'] = row.get('delta', 0)
        metrics['gamma'] = row.get('gamma', 0)
        metrics['theta'] = row.get('theta', 0)
        metrics['vega'] = row.get('vega', 0)

        # Ликвидность (упрощенная)
        if strike > 0:
            metrics['liquidity_score'] = (volume * oi) / (strike * 1000)  # нормализация
        else:
            metrics['liquidity_score'] = 0

        # Отношение Volume/OI (активность)
        metrics['vol_oi_ratio'] = volume / oi if oi > 0 else 0

        # IV Percentile (требует исторических данных, здесь заглушка)
        metrics['iv_rank'] = self._calculate_iv_rank(row.get('instrument_name', ''), iv)

        # Мoneyness (в деньгах/из денег)
        if underlying_price > 0:
            metrics['moneyness'] = (strike / underlying_price) - 1  # -0.1 = 10% OTM put
            metrics['in_the_money'] = (
                (strike < underlying_price)
                if row.get('option_type') == 'call'
                else (strike > underlying_price)
            )
        else:
            metrics['moneyness'] = 0
            metrics['in_the_money'] = False

        # Риск/прибыль (приближенно, для 1 контракта long/short)
        mark_price = row.get('mark_price', 0)
        premium_quote = mark_price * underlying_price if mark_price <= 1 else mark_price
        premium_quote = float(max(premium_quote, 0))
        option_type = row.get('option_type')

        metrics['premium_quote'] = premium_quote
        metrics['long_max_loss'] = premium_quote
        metrics['short_max_profit'] = premium_quote

        if option_type == 'call':
            metrics['long_max_profit'] = float('inf')
            metrics['short_max_loss'] = float('inf')
        else:
            metrics['long_max_profit'] = float(max(strike - premium_quote, 0))
            metrics['short_max_loss'] = float(max(strike - premium_quote, 0))

        return metrics

    def _calculate_iv_rank(self, instrument: str, current_iv: float) -> float:
        """Расчет IV Rank (0-100) на основе исторических данных"""
        # В реальном использовании здесь должен быть запрос к БД (Redis/PostgreSQL)
        # с историей IV за 30-90 дней
        hist = self.iv_history.get(instrument, [])
        if len(hist) < 30:
            return 50.0  # Нейтральное значение при недостатке данных

        iv_min = np.min(hist)
        iv_max = np.max(hist)
        if iv_max == iv_min:
            return 50.0

        current_iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        return float(min(max(current_iv_rank, 0), 100))

    def scan(self, filters: OptionFilters = None) -> pd.DataFrame:
        """
        Основной метод сканирования
        Возвращает DataFrame с отфильтрованными опционами
        """
        if filters is None:
            filters = OptionFilters()

        print(f"🔍 Сканирование опционов {filters.currency}...")

        # 1. Получаем список инструментов
        instruments = self.get_instruments(filters.currency, expired=False)
        if not instruments:
            print("Ошибка получения списка инструментов")
            return pd.DataFrame()

        # Фильтр по perpetual (исключаем если нужно)
        if filters.exclude_perpetual:
            instruments = [i for i in instruments if not i['name'].endswith('-PERPETUAL')]

        # Фильтр по типу пары (инверсная / неинверсная)
        if filters.instrument_type in {"inverse", "non_inverse"}:
            instruments = [
                i for i in instruments
                if self._parse_instrument_market(i["name"])["pair_type"] == filters.instrument_type
            ]

        instrument_names = [i['name'] for i in instruments]
        print(f"Найдено инструментов: {len(instrument_names)}")

        # 2. Получаем рыночные данные
        tickers = self.get_ticker_batch(instrument_names)

        # 3. Получаем цену базового актива (index price)
        index_price = self._get_index_price(filters.currency)

        # 4. Обработка данных
        results = []
        for inst in instruments:
            name = inst['name']
            ticker = tickers.get(name)
            if not ticker:
                continue

            # Базовые данные
            row_data = {
                'instrument_name': name,
                'option_type': inst['option_type'],
                'strike': inst['strike'],
                'expiration_timestamp': inst['expiration_timestamp'],
                'dte': self.calculate_dte(inst['expiration_timestamp']),
                'mark_price': ticker.get('mark_price', 0),
                'bid_price': ticker.get('bid_price', 0),
                'ask_price': ticker.get('ask_price', 0),
                'volume': ticker.get('volume', 0),
                'open_interest': ticker.get('open_interest', 0),
                'iv': ticker.get('iv', 0),
                'underlying_price': index_price
            }

            market_info = self._parse_instrument_market(name)
            row_data.update(market_info)

            # Расширенные метрики (передаём ticker + нужные поля)
            ticker_with_meta = dict(ticker)
            ticker_with_meta['instrument_name'] = name
            ticker_with_meta['option_type'] = inst['option_type']
            ticker_with_meta['strike'] = inst['strike']
            metrics = self.calculate_metrics(ticker_with_meta, index_price)
            row_data.update(metrics)

            # Применяем фильтры
            if not self._apply_filters(row_data, filters, index_price):
                continue

            # Спред (ликвидность)
            bid = row_data['bid_price']
            ask = row_data['ask_price']
            mark = row_data['mark_price']
            if bid and ask:
                row_data['spread_pct'] = (ask - bid) / mark * 100 if mark > 0 else 100
            else:
                row_data['spread_pct'] = 100  # Неликвидный

            results.append(row_data)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('liquidity_score', ascending=False)
            print(f"✅ Найдено подходящих опционов: {len(df)}")

        return df

    def _apply_filters(self, data: dict, filters: OptionFilters, underlying_price: float) -> bool:
        """Применение фильтров"""
        # IV фильтр
        if not (filters.iv_min <= data['iv'] <= filters.iv_max):
            return False

        # Delta фильтр (если есть)
        delta = data.get('delta')
        if delta is not None:
            if not (filters.delta_min <= delta <= filters.delta_max):
                return False
        else:
            # Если дельты нет, пропускаем (может быть для deep OTM)
            return False

        # DTE фильтр
        if not (filters.dte_min <= data['dte'] <= filters.dte_max):
            return False

        # Ликвидность
        if data['volume'] < filters.min_volume:
            return False
        if data['open_interest'] < filters.min_open_interest:
            return False
        if data['liquidity_score'] < filters.min_liquidity_score:
            return False

        # Цена базового актива
        low, high = filters.underlying_price_range
        if not (low <= underlying_price <= high):
            return False

        return True

    def _get_index_price(self, currency: str) -> float:
        """Получить текущую цену индекса (BTC/USD или ETH/USD)"""
        result = self._get("/public/get_index", {'index_name': f"{currency}_USD"})
        return result.get('index_price', 0) if isinstance(result, dict) else 0

    # ==================== СТРАТЕГИЧЕСКИЕ СКАНЕРЫ ====================

    def scan_high_iv(self, currency: str = "BTC", iv_threshold: float = 0.8) -> pd.DataFrame:
        """
        Поиск опционов с высокой IV (для стратегии продажи волатильности)
        IV Rank > 80% означает, что текущая волатильность выше 80% времени за последние 90 дней
        """
        filters = OptionFilters(
            currency=currency,
            min_volume=1.0,
            min_open_interest=50.0,
            iv_min=0.5,  # минимум 50% IV
            dte_min=7,   # минимум неделя до экспирации
            dte_max=60   # максимум 2 месяца (для theta decay)
        )

        df = self.scan(filters)
        if df.empty:
            return df

        # Фильтруем по IV Rank (если есть исторические данные)
        high_iv = df[df['iv_rank'] >= (iv_threshold * 100)]
        return high_iv.sort_values('iv_rank', ascending=False)

    def scan_iron_condor_setup(self, currency: str = "BTC") -> pd.DataFrame:
        """
        Поиск опционов для Iron Condor (продажа спредов в зоне безубытка)
        Ищем ликвидные опционы с 30-45 DTE и умеренной IV
        """
        filters = OptionFilters(
            currency=currency,
            min_volume=5.0,
            min_open_interest=100.0,
            dte_min=30,
            dte_max=45,
            iv_min=0.3,
            iv_max=0.7  # не слишком высокая, не слишком низкая
        )

        df = self.scan(filters)
        if df.empty:
            return df

        # Добавляем расчет потенциальной прибыли (упрощенно)
        df = df.copy()
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        df['expected_theta'] = df['theta'] * 30  # 30 дней тета-декая

        return df.sort_values('liquidity_score', ascending=False)

    def scan_arbitrage_opportunities(self, currency: str = "BTC") -> List[dict]:
        """
        Поиск арбитража Put-Call Parity (базовый)
        C - P = S - K*e^(-rT)
        На практике проверяем разницу между синтетическим фьючерсом и реальным
        """
        print("🔍 Сканирование арбитражных возможностей...")

        # Получаем все опционы одной экспирации
        instruments = self.get_instruments(currency)
        index_price = self._get_index_price(currency)

        # Группируем по экспирации
        exp_groups: Dict[int, Dict[str, list]] = {}
        for inst in instruments:
            exp = inst['expiration_timestamp']
            if exp not in exp_groups:
                exp_groups[exp] = {'calls': [], 'puts': []}

            if inst['option_type'] == 'call':
                exp_groups[exp]['calls'].append(inst)
            else:
                exp_groups[exp]['puts'].append(inst)

        opportunities = []

        for exp, group in list(exp_groups.items()):
            if len(group['calls']) < 3 or len(group['puts']) < 3:
                continue

            # Берем ATM опционы (closest to spot)
            calls = sorted(group['calls'], key=lambda x: abs(x['strike'] - index_price))
            puts = sorted(group['puts'], key=lambda x: abs(x['strike'] - index_price))

            for call, put in zip(calls[:5], puts[:5]):  # Топ-5 ATM
                if call['strike'] != put['strike']:
                    continue

                # Получаем тикеры
                call_ticker = self.get_ticker_batch([call['name']]).get(call['name'], {})
                put_ticker = self.get_ticker_batch([put['name']]).get(put['name'], {})

                if not call_ticker or not put_ticker:
                    continue

                call_price = (call_ticker.get('bid_price', 0) + call_ticker.get('ask_price', 0)) / 2
                put_price = (put_ticker.get('bid_price', 0) + put_ticker.get('ask_price', 0)) / 2

                if call_price == 0 or put_price == 0:
                    continue

                # Синтетический фьючерс (Call - Put)
                synthetic_future = call_price - put_price
                strike = call['strike']

                # Теоретическая стоимость (упрощенно, без ставки)
                dte = self.calculate_dte(exp)
                theoretical = index_price - strike  # без дисконтирования для простоты

                # Арбитраж: если синтетический сильно отклоняется от спота
                arb_size = abs(synthetic_future - theoretical)
                arb_pct = arb_size / index_price * 100 if index_price > 0 else 0

                if arb_pct > 0.5:  # 0.5% отклонение - потенциальная возможность
                    opportunities.append({
                        'expiration': exp,
                        'strike': strike,
                        'dte': dte,
                        'call_price': call_price,
                        'put_price': put_price,
                        'synthetic_future': synthetic_future,
                        'spot': index_price,
                        'arb_size': arb_size,
                        'arb_pct': arb_pct,
                        'type': 'synthetic_future_vs_spot'
                    })

        return sorted(opportunities, key=lambda x: x['arb_pct'], reverse=True)

    def export_to_csv(self, df: pd.DataFrame, filename: str = "options_scan.csv") -> None:
        """Экспорт результатов"""
        df.to_csv(filename, index=False)
        print(f"📊 Данные экспортированы в {filename}")


# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

if __name__ == "__main__":
    # Инициализация (без ключей для публичных данных, с ключами для приватных)
    scanner = DeribitOptionsScanner(
        api_key=os.getenv("DERIBIT_API_KEY"),
        api_secret=os.getenv("DERIBIT_API_SECRET")
    )

    # 1. Общий скан с фильтрами
    print("\n=== Общий скан ликвидных опционов BTC ===")
    filters = OptionFilters(
        currency="BTC",
        min_volume=2.0,
        min_open_interest=20.0,
        iv_min=0.2,
        iv_max=1.5,
        dte_min=7,
        dte_max=90,
        delta_min=-0.5,
        delta_max=0.5  # Дельта между -0.5 и 0.5 (ATM зона)
    )

    results = scanner.scan(filters)

    if not results.empty:
        # Вывод топ-10 по ликвидности
        print(results[['instrument_name', 'option_type', 'strike', 'dte', 'iv', 'delta',
                        'volume', 'open_interest', 'liquidity_score', 'iv_rank']].head(10))

        # Сохранение
        scanner.export_to_csv(results, "btc_options_scan.csv")

    # 2. Поиск высокой волатильности (для продажи)
    print("\n=== Высокая IV (продажа волатильности) ===")
    high_iv = scanner.scan_high_iv(currency="BTC", iv_threshold=0.85)
    if not high_iv.empty:
        print(high_iv[['instrument_name', 'iv', 'iv_rank', 'dte', 'strike', 'delta']].head(10))

    # 3. Iron Condor настройки
    print("\n=== Iron Condor setups ===")
    ic = scanner.scan_iron_condor_setup(currency="BTC")
    if not ic.empty:
        print(ic[['instrument_name', 'dte', 'iv', 'theta', 'liquidity_score']].head(10))

    # 4. Арбитраж (требует быстрого исполнения!)
    print("\n=== Арбитражные возможности (Put-Call Parity) ===")
    arb = scanner.scan_arbitrage_opportunities(currency="BTC")
    for opp in arb[:5]:
        print(f"Strike: {opp['strike']}, DTE: {opp['dte']}, "
              f"Arb: {opp['arb_pct']:.2f}%, Type: {opp['type']}")
