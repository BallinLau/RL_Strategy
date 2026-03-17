"""
市场状态识别
Market State Identification
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class MarketStateIdentifier:
    """
    市场状态识别器：识别5种市场状态
    """
    
    def __init__(self, fast_period: int = 5, slow_period: int = 20, 
                 lookback_periods: int = 3, threshold_score: int = 2):
        """
        初始化市场状态识别器
        
        Args:
            fast_period: 快速动量周期
            slow_period: 慢速动量周期
            lookback_periods: 回看周期数
            threshold_score: 持续性判断阈值
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.lookback_periods = lookback_periods
        self.threshold_score = threshold_score
    
    def calculate_momentum(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        计算快慢速动量信号
        
        Args:
            prices: 价格序列
        
        Returns:
            momentum_dict: 动量信号字典
        """
        fast_momentum = (prices / prices.shift(self.fast_period)) - 1
        slow_momentum = (prices / prices.shift(self.slow_period)) - 1
        
        return {
            'fast_momentum': fast_momentum,
            'slow_momentum': slow_momentum
        }
    
    def calculate_persistence_score(self, momentum: pd.Series) -> pd.Series:
        """
        计算动量持续性评分
        
        Args:
            momentum: 动量序列
        
        Returns:
            scores: 持续性评分序列
        """
        scores = pd.Series(0, index=momentum.index)
        
        for i in range(self.lookback_periods, len(momentum)):
            recent_values = momentum.iloc[i-self.lookback_periods:i]
            
            # 基础分：正值个数
            positive_count = (recent_values > 0).sum()
            score = positive_count
            
            # 额外分：连续递增
            is_increasing = all(recent_values.iloc[j] < recent_values.iloc[j+1] 
                               for j in range(len(recent_values)-1))
            if is_increasing:
                score += 1
            
            scores.iloc[i] = score
        
        return scores
    
    def identify_market_state(self, prices: pd.Series) -> pd.Series:
        """
        识别市场状态
        
        Args:
            prices: 价格序列
        
        Returns:
            states: 市场状态序列
        """
        momentum = self.calculate_momentum(prices)
        fast_mom = momentum['fast_momentum']
        slow_mom = momentum['slow_momentum']
        
        fast_score = self.calculate_persistence_score(fast_mom)
        slow_score = self.calculate_persistence_score(slow_mom)
        
        states = pd.Series('SIDEWAYS', index=prices.index)
        
        for i in range(self.slow_period, len(prices)):
            fast_val = fast_mom.iloc[i]
            slow_val = slow_mom.iloc[i]
            fast_s = fast_score.iloc[i]
            slow_s = slow_score.iloc[i]
            
            # 牛市：快慢动量均为正且持续向上
            if (fast_val > 0 and slow_val > 0 and 
                fast_s >= self.threshold_score and slow_s >= self.threshold_score):
                states.iloc[i] = 'BULL'
            
            # 熊市：快慢动量均为负且持续向下
            elif (fast_val < 0 and slow_val < 0 and 
                  fast_s >= self.threshold_score and slow_s >= self.threshold_score):
                states.iloc[i] = 'BEAR'
            
            # 修正市：慢速为正但快速转负
            elif slow_val > 0 and fast_val < 0 and slow_s >= self.threshold_score:
                states.iloc[i] = 'CORRECTION'
            
            # 反弹市：慢速为负但快速转正
            elif slow_val < 0 and fast_val > 0 and slow_s >= self.threshold_score:
                states.iloc[i] = 'REBOUND'
            
            # 其他情况：震荡市
            else:
                states.iloc[i] = 'SIDEWAYS'
        
        return states


class VolatilityCalculator:
    """
    波动率计算器：使用EWMA方法
    """
    
    def __init__(self, lambda_decay: float = 0.94):
        """
        初始化波动率计算器
        
        Args:
            lambda_decay: EWMA衰减因子
        """
        self.lambda_decay = lambda_decay
        self.last_variance = None
    
    def calculate_ewma_volatility(self, returns: pd.Series) -> pd.Series:
        """
        计算EWMA波动率
        
        Args:
            returns: 收益率序列
        
        Returns:
            volatility: 波动率序列
        """
        variances = pd.Series(0.0, index=returns.index)
        
        # 初始化：使用前20个周期的样本方差
        if len(returns) > 20:
            variances.iloc[19] = returns.iloc[:20].var()
        
        # 递推计算
        for i in range(20, len(returns)):
            variances.iloc[i] = (self.lambda_decay * variances.iloc[i-1] + 
                                (1 - self.lambda_decay) * returns.iloc[i]**2)
        
        volatility = np.sqrt(variances)
        return volatility
    
    def classify_volatility_regime(self, volatility: pd.Series, 
                                   high_threshold: float = 0.02, 
                                   low_threshold: float = 0.01) -> pd.Series:
        """
        分类波动率状态
        
        Args:
            volatility: 波动率序列
            high_threshold: 高波动阈值
            low_threshold: 低波动阈值
        
        Returns:
            regime: 波动率状态序列
        """
        regime = pd.Series('MEDIUM', index=volatility.index)
        regime[volatility > high_threshold] = 'HIGH'
        regime[volatility < low_threshold] = 'LOW'
        return regime


class MarketStateManager:
    """
    市场状态管理器：整合市场状态识别和波动率计算
    """
    
    def __init__(self):
        self.state_identifier = MarketStateIdentifier()
        self.volatility_calculator = VolatilityCalculator()
        self.current_state = None
        self.current_volatility_regime = None
    
    def update(self, prices: pd.Series) -> Dict[str, Any]:
        """
        更新市场状态
        
        Args:
            prices: 价格序列
        
        Returns:
            state_info: 市场状态信息字典
        """
        # 识别市场状态
        states = self.state_identifier.identify_market_state(prices)
        self.current_state = states.iloc[-1]
        
        # 计算波动率
        returns = prices.pct_change()
        volatility = self.volatility_calculator.calculate_ewma_volatility(returns)
        vol_regime = self.volatility_calculator.classify_volatility_regime(volatility)
        self.current_volatility_regime = vol_regime.iloc[-1]
        
        # 获取最新动量值
        momentum = self.state_identifier.calculate_momentum(prices)
        
        return {
            'state': self.current_state,
            'volatility_regime': self.current_volatility_regime,
            'fast_momentum': momentum['fast_momentum'].iloc[-1],
            'slow_momentum': momentum['slow_momentum'].iloc[-1],
            'volatility': volatility.iloc[-1]
        }
    
    def get_current_state(self) -> str:
        """获取当前市场状态"""
        return self.current_state
    
    def get_current_volatility_regime(self) -> str:
        """获取当前波动率状态"""
        return self.current_volatility_regime
