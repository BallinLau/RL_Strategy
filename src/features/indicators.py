"""
技术指标计算器
Technical Indicators Calculator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class IndicatorCalculator:
    """
    技术指标计算器：计算各种技术指标
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化指标计算器
        
        Args:
            df: 包含OHLCV的DataFrame（20分钟周期）
        """
        self.df = df.copy()
        self.indicators: Dict[str, pd.Series] = {}
    
    def calculate_all(self) -> Dict[str, pd.Series]:
        """
        计算所有指标（文档要求的39个指标）
        
        Returns:
            indicators: 所有指标的字典
        """
        # 1. 移动平均线（MA5, MA10, MA20, MA60）
        self.indicators.update(self.calculate_ma())
        
        # 2. 布林带指标
        self.indicators.update(self.calculate_bollinger_bands())
        
        # 3. MACD指标
        self.indicators.update(self.calculate_macd())
        
        # 4. ADX及相关指标
        self.indicators.update(self.calculate_adx())
        
        # 5. ATR（平均真实波幅）
        self.indicators['ATR'] = self.calculate_atr()
        
        # 6. RSI（相对强弱指标）
        self.indicators['RSI'] = self.calculate_rsi()
        
        # 7. 成交量指标
        self.indicators.update(self.calculate_volume_indicators())
        
        # 8. 动量指标
        self.indicators.update(self.calculate_momentum_indicators())
        
        # 9. 多因子模型指标（价值、动量、质量因子等）
        self.indicators.update(self.calculate_factor_model_indicators())
        
        # 10. 简化的配对交易指标（使用技术指标模拟）
        self.indicators.update(self.calculate_simplified_pair_trading_indicators())
        
        # print(f"已计算 {len(self.indicators)} 个技术指标")  # 注释掉以提升性能
        return self.indicators
    
    def get_indicator(self, name: str) -> Optional[pd.Series]:
        """
        获取指定指标
        
        Args:
            name: 指标名称
        
        Returns:
            indicator: 指标序列
        """
        return self.indicators.get(name)
    
    def calculate_ma(self, periods: List[int] = [5, 10, 20, 60]) -> Dict[str, pd.Series]:
        """
        计算移动平均线
        
        Args:
            periods: 周期列表
        
        Returns:
            ma_dict: MA指标字典
        """
        results = {}
        for period in periods:
            results[f'MA{period}'] = self.df['close'].rolling(window=period).mean()
        return results
    
    def calculate_ema(self, periods: List[int] = [12, 26]) -> Dict[str, pd.Series]:
        """
        计算指数移动平均线
        
        Args:
            periods: 周期列表
        
        Returns:
            ema_dict: EMA指标字典
        """
        results = {}
        for period in periods:
            results[f'EMA{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
        return results

    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        计算布林带
        
        Args:
            period: 周期
            std_dev: 标准差倍数
        
        Returns:
            bb_dict: 布林带指标字典
        """
        close = self.df['close']
        
        # 中轨（MA20）
        middle = close.rolling(window=period).mean()
        
        # 标准差
        std = close.rolling(window=period).std()
        
        # 上轨和下轨
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        # 带宽
        width = (upper - lower) / (middle + 1e-8)
        
        # %b指标
        percent_b = (close - lower) / ((upper - lower) + 1e-8)
        
        return {
            'BB_MIDDLE': middle.fillna(0),
            'BB_UPPER': upper.fillna(0),
            'BB_LOWER': lower.fillna(0),
            'BB_WIDTH': width.fillna(0),
            'BB_PERCENT_B': percent_b.fillna(0)
        }
    
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, 
                       signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        计算MACD指标
        
        Args:
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
        
        Returns:
            macd_dict: MACD指标字典
        """
        close = self.df['close']
        
        # 计算EMA
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        
        # 计算DIF和DEA
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal_period, adjust=False).mean()
        hist = dif - dea
        
        # 检测金叉死叉
        cross = pd.Series(0, index=dif.index)
        cross[(dif > dea) & (dif.shift(1) <= dea.shift(1))] = 1  # 金叉
        cross[(dif < dea) & (dif.shift(1) >= dea.shift(1))] = -1  # 死叉
        
        return {
            'MACD_DIF': dif.fillna(0),
            'MACD_DEA': dea.fillna(0),
            'MACD_HIST': hist.fillna(0),
            'MACD_CROSS': cross.fillna(0)
        }
    
    def calculate_adx(self, period: int = 14) -> Dict[str, pd.Series]:
        """
        计算ADX及相关指标
        
        Args:
            period: 周期
        
        Returns:
            adx_dict: ADX指标字典
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # 计算TR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算+DM和-DM
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low
        
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        
        plus_dm[(high_diff > 0) & (high_diff > low_diff)] = high_diff
        minus_dm[(low_diff > 0) & (low_diff > high_diff)] = low_diff
        
        # 计算+DI和-DI
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-8)
        
        # 计算ADX
        dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return {
            'ADX': adx.fillna(0),
            'PLUS_DI': plus_di.fillna(0),
            'MINUS_DI': minus_di.fillna(0)
        }
    
    def calculate_atr(self, period: int = 14) -> pd.Series:
        """
        计算ATR（平均真实波幅）
        
        Args:
            period: 周期
        
        Returns:
            atr: ATR序列
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        计算RSI（相对强弱指标）
        
        Args:
            period: 周期
        
        Returns:
            rsi: RSI序列
        """
        close = self.df['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_pair_trading_indicators(self, stock_a: pd.Series, stock_b: pd.Series, 
                                           window: int = 20) -> Dict[str, pd.Series]:
        """
        计算配对交易相关指标（文档要求的功能）
        
        Args:
            stock_a: 股票A的价格序列
            stock_b: 股票B的价格序列
            window: 滚动窗口大小
        
        Returns:
            pair_dict: 配对交易指标字典
        """
        # 计算滚动Beta（组合Beta值）
        cov = stock_a.rolling(window=window).cov(stock_b)
        var = stock_b.rolling(window=window).var()
        beta = cov / var
        
        # 计算价差
        spread = stock_a - beta * stock_b
        
        # 计算Z-score
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        spread_zscore = (spread - spread_mean) / spread_std
        
        # 计算滚动相关系数
        correlation = stock_a.rolling(window=window).corr(stock_b)
        
        # 估计半衰期
        half_life = self._estimate_half_life(spread, window)
        
        # 计算协整性检验结果（ADF统计量）
        adf_stats = self._calculate_rolling_adf(spread, window)
        
        # 计算成交量比（假设有成交量数据）
        volume_ratio = self._calculate_volume_ratio(stock_a, stock_b)
        
        return {
            'SPREAD': spread.fillna(0),
            'SPREAD_ZSCORE': spread_zscore.fillna(0),
            'SPREAD_MEAN': spread_mean.fillna(0),
            'SPREAD_STD': spread_std.fillna(0),
            'CORRELATION': correlation.fillna(0),
            'BETA': beta.fillna(0),
            'HALF_LIFE': half_life.fillna(0),
            'ADF_STAT': adf_stats.fillna(0),  # 协整性检验结果
            'VOLUME_RATIO': volume_ratio.fillna(0)  # 成交量比
        }
    
    def _estimate_half_life(self, series: pd.Series, window: int) -> pd.Series:
        """
        估计价差半衰期
        
        Args:
            series: 价差序列
            window: 窗口大小
        
        Returns:
            half_life: 半衰期序列
        """
        half_lives = []
        
        for i in range(window, len(series)):
            y = series.iloc[i-window:i].diff().dropna()
            x = series.iloc[i-window:i].shift(1).dropna()
            
            if len(x) > 0 and len(y) > 0:
                x = x.iloc[:len(y)]
                if x.std() > 0:
                    # 简单线性回归估计自回归系数
                    rho = np.corrcoef(x, y)[0, 1] * (y.std() / x.std())
                    if 0 < rho < 1:
                        half_life = -np.log(2) / np.log(rho)
                        half_lives.append(half_life)
                    else:
                        half_lives.append(np.nan)
                else:
                    half_lives.append(np.nan)
            else:
                half_lives.append(np.nan)
        
        # 填充前面的NaN
        result = pd.Series([np.nan] * window + half_lives, index=series.index)
        return result
    
    def calculate_volume_indicators(self) -> Dict[str, pd.Series]:
        """
        计算成交量相关指标
        
        Returns:
            volume_dict: 成交量指标字典
        """
        volume = self.df['volume']
        
        # 成交量移动平均
        volume_ma5 = volume.rolling(window=5).mean()
        volume_ma10 = volume.rolling(window=10).mean()
        
        # 成交量比率
        volume_ratio = volume / (volume_ma5 + 1e-8)
        
        return {
            'VOLUME_MA5': volume_ma5.fillna(0),
            'VOLUME_MA10': volume_ma10.fillna(0),
            'VOLUME_RATIO': volume_ratio.fillna(0)
        }
    
    def _calculate_rolling_adf(self, series: pd.Series, window: int) -> pd.Series:
        """
        计算滚动ADF统计量（协整性检验）
        
        Args:
            series: 价差序列
            window: 窗口大小
        
        Returns:
            adf_stats: ADF统计量序列
        """
        from statsmodels.tsa.stattools import adfuller
        
        adf_values = []
        
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i].dropna()
            if len(window_data) >= 10:  # 需要足够的数据
                try:
                    adf_result = adfuller(window_data, autolag='AIC')
                    adf_values.append(adf_result[0])  # ADF统计量
                except:
                    adf_values.append(np.nan)
            else:
                adf_values.append(np.nan)
        
        # 填充前面的NaN
        result = pd.Series([np.nan] * window + adf_values, index=series.index)
        return result
    
    def _calculate_volume_ratio(self, stock_a: pd.Series, stock_b: pd.Series) -> pd.Series:
        """
        计算成交量比
        
        Args:
            stock_a: 股票A的价格序列
            stock_b: 股票B的价格序列
        
        Returns:
            volume_ratio: 成交量比序列
        """
        # 这里假设有成交量数据，实际应用中需要传入成交量序列
        # 简化实现：使用价格波动作为代理
        vol_a = stock_a.pct_change().rolling(window=20).std()
        vol_b = stock_b.pct_change().rolling(window=20).std()
        
        volume_ratio = vol_a / vol_b
        return volume_ratio
    
    def calculate_factor_model_indicators(self) -> Dict[str, pd.Series]:
        """
        计算多因子模型指标（文档要求的功能）
        
        包括：
        1. 多因子综合得分（价值、动量、质量因子）
        2. 股票相对基准指数的超额收益
        3. 行业暴露度
        4. 个股波动率
        
        Returns:
            factor_dict: 多因子指标字典
        """
        close = self.df['close']
        
        # 1. 个股波动率（年化）
        daily_returns = close.pct_change()
        annual_volatility = daily_returns.rolling(window=20).std() * np.sqrt(252)
        
        # 2. 价值因子（简化：市盈率倒数）
        # 实际应用中需要财务数据，这里用价格/均线比作为代理
        value_factor = close / close.rolling(window=60).mean()
        
        # 3. 动量因子（过去20周期收益率）
        momentum_factor = close.pct_change(periods=20)
        
        # 4. 质量因子（简化：波动率倒数）
        quality_factor = 1 / annual_volatility
        
        # 5. 多因子综合得分（等权重）
        # 标准化各因子
        def standardize(series):
            return (series - series.mean()) / series.std()
        
        value_norm = standardize(value_factor)
        momentum_norm = standardize(momentum_factor)
        quality_norm = standardize(quality_factor)
        
        # 综合得分
        composite_score = (value_norm + momentum_norm + quality_norm) / 3
        
        # 6. 超额收益（相对于60日均线）
        ma_60 = close.rolling(window=60).mean()
        excess_return = close.pct_change() - ma_60.pct_change()
        # 填充nan值
        excess_return = excess_return.fillna(0)
        
        # 7. 行业暴露度（简化：价格相对移动平均的标准化偏离）
        # 避免使用复杂的相关性计算，改用简单的标准化偏离度
        ma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        market_correlation = ((close - ma_20) / (std_20 + 1e-8)).fillna(0)
        
        return {
            'VOLATILITY': annual_volatility.fillna(0),
            'VALUE_FACTOR': value_factor.fillna(0),
            'MOMENTUM_FACTOR': momentum_factor.fillna(0),
            'QUALITY_FACTOR': quality_factor.fillna(0),
            'COMPOSITE_SCORE': composite_score.fillna(0),
            'EXCESS_RETURN': excess_return,
            'INDUSTRY_EXPOSURE': market_correlation
        }
    
    def calculate_simplified_pair_trading_indicators(self) -> Dict[str, pd.Series]:
        """
        计算简化的配对交易指标（使用技术指标模拟）
        
        由于没有真实的股票对数据，使用技术指标构造模拟的配对交易指标
        这样MarketNeutral和StatisticalArbitrage策略就能正常工作
        
        Returns:
            pair_dict: 配对交易指标字典
        """
        close = self.df['close']
        
        # 1. 模拟价差（使用价格与MA20的差值）
        ma20 = close.rolling(window=20).mean()
        spread = close - ma20
        
        # 2. 计算价差的统计特征
        spread_mean = spread.rolling(window=20).mean()
        spread_std = spread.rolling(window=20).std()
        
        # 3. 计算Z-score
        spread_zscore = (spread - spread_mean) / (spread_std + 1e-8)
        
        # 4. 模拟相关系数（使用RSI与50的距离模拟）
        rsi = self.indicators.get('RSI', pd.Series([50] * len(close), index=close.index))
        correlation = 1.0 - abs(rsi - 50) / 50  # RSI越接近50，"相关性"越高
        
        # 5. 模拟Beta（使用简化计算，避免慢循环）
        returns = close.pct_change()
        market_returns = ma20.pct_change()  # 使用MA20变化率作为"市场"收益
        
        # 使用滚动相关系数和波动率比来快速估算Beta
        rolling_corr = returns.rolling(window=20).corr(market_returns)
        returns_std = returns.rolling(window=20).std()
        market_std = market_returns.rolling(window=20).std()
        
        # Beta = 相关系数 * (股票波动率 / 市场波动率)
        beta = rolling_corr * (returns_std / (market_std + 1e-8))
        beta = beta.fillna(1.0)
        
        # 6. 模拟半衰期（使用ATR倒数）
        atr = self.indicators.get('ATR', pd.Series([1.0] * len(close), index=close.index))
        half_life = 20 / (atr + 0.01)  # ATR越大，半衰期越短
        
        # 7. 模拟ADF统计量（使用趋势强度）
        adx = self.indicators.get('ADX', pd.Series([25] * len(close), index=close.index))
        adf_stat = -adx / 10  # ADX越高，ADF统计量越负（越显著）
        
        # 8. 成交量比（使用现有的VOLUME_RATIO）
        volume_ratio = self.indicators.get('VOLUME_RATIO', pd.Series([1.0] * len(close), index=close.index))
        
        return {
            'SPREAD': spread.fillna(0),
            'SPREAD_ZSCORE': spread_zscore.fillna(0),
            'SPREAD_MEAN': spread_mean.fillna(0),
            'SPREAD_STD': spread_std.fillna(1),
            'CORRELATION': correlation.fillna(0.5),
            'BETA': beta.fillna(1.0),
            'HALF_LIFE': half_life.fillna(20),
            'ADF_STAT': adf_stat.fillna(-2.5),
            'VOLUME_RATIO': volume_ratio.fillna(1.0)
        }
    
    def calculate_momentum_indicators(self) -> Dict[str, pd.Series]:
        """
        计算动量指标
        
        Returns:
            momentum_dict: 动量指标字典
        """
        close = self.df['close']
        
        # ROC (Rate of Change)
        roc = close.pct_change(periods=10) * 100
        
        # Momentum
        momentum = close - close.shift(10)
        
        return {
            'ROC': roc.fillna(0),
            'MOMENTUM': momentum.fillna(0)
        }
