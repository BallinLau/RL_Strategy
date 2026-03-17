"""
股票多空策略

基于多因子模型的股票多空策略
"""

import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy


class LongShortEquityStrategy(BaseStrategy):
    """股票多空策略（文档要求的功能）"""
    
    def __init__(self, top_percentile: float = 0.1, bottom_percentile: float = 0.1,
                 rebalance_period: int = 20, industry_neutral_weight: float = 0.3,
                 market_neutral_weight: float = 0.3):
        """
        初始化股票多空策略（增强版）
        
        Args:
            top_percentile: 多头组合百分位（前10%）
            bottom_percentile: 空头组合百分位（后10%）
            rebalance_period: 再平衡周期（20分钟）
            industry_neutral_weight: 行业中性权重
            market_neutral_weight: 市场中性权重
        """
        super().__init__("LongShortEquity")
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
        self.rebalance_period = rebalance_period
        self.industry_neutral_weight = industry_neutral_weight
        self.market_neutral_weight = market_neutral_weight
        
        # 设置策略类型为趋势策略
        self.strategy_type = 'TREND'
        
        # 跟踪状态
        self.last_rebalance_time = 0
        self.current_portfolio = {}
    
    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成股票多空信号（文档要求的功能）
        
        文档要求：
        1. 信号强度 = 预期超额收益 × 组合的风险中性程度
        2. 使用多因子模型对股票打分排序
        3. 买入排名前10%的股票（多头），卖出排名后10%的股票（空头）
        4. 保持行业、市值中性，定期再平衡
        5. 根据波动率状态调节信号强度
        """
        indicators = state.get('indicators', {})
        market_state = state.get('market_state', {})
        current_time = state.get('current_time', 0)
        
        # 获取多因子模型指标
        composite_score = indicators.get('COMPOSITE_SCORE', 0.5)
        excess_return = indicators.get('EXCESS_RETURN', 0)
        industry_exposure = indicators.get('INDUSTRY_EXPOSURE', 0.5)
        volatility = indicators.get('VOLATILITY', 0.2)
        
        # 计算预期超额收益成分
        expected_excess_return = excess_return
        
        # 计算风险中性程度成分（文档要求）
        # 行业暴露度越接近0.5表示越中性
        industry_neutrality = 1.0 - abs(industry_exposure - 0.5) * 2
        
        # 计算原始信号强度（文档要求：预期超额收益 × 风险中性程度）
        excess_return_component = min(abs(expected_excess_return) / 0.1, 1.0)  # 假设10%为最大预期超额收益
        risk_neutral_component = industry_neutrality
        
        original_strength = excess_return_component * risk_neutral_component
        original_strength = min(original_strength, 1.0)
        
        # 根据波动率状态调节信号强度
        adjusted_strength = self.adjust_signal_by_volatility(original_strength, market_state)
        
        # 检查是否需要再平衡（文档要求：定期再平衡）
        need_rebalance = (current_time - self.last_rebalance_time) >= self.rebalance_period
        
        # 初始化信号
        signal_strength = adjusted_strength
        direction = 'NEUTRAL'
        confidence = 0.0
        metadata = {
            'composite_score': composite_score,
            'excess_return': excess_return,
            'industry_exposure': industry_exposure,
            'volatility': volatility,
            'expected_excess_return': expected_excess_return,
            'industry_neutrality': industry_neutrality,
            'excess_return_component': excess_return_component,
            'risk_neutral_component': risk_neutral_component,
            'original_strength': original_strength,
            'adjusted_strength': adjusted_strength,
            'volatility_regime': market_state.get('volatility_regime', 'MEDIUM'),
            'need_rebalance': need_rebalance,
            'current_portfolio': self.current_portfolio
        }
        
        # 多头信号（高分股票，前10%）
        if composite_score > (1 - self.top_percentile):
            signal_strength = (composite_score - (1 - self.top_percentile)) / self.top_percentile
            signal_strength = min(signal_strength, 1.0)
            direction = 'LONG'
            
            # 置信度基于超额收益和行业中性程度
            excess_confidence = min(abs(expected_excess_return) / 0.05, 1.0)  # 5%超额收益为高置信度
            neutral_confidence = industry_neutrality
            confidence = signal_strength * (excess_confidence * 0.6 + neutral_confidence * 0.4)
            confidence = max(confidence, 0.1)  # 确保置信度不为负数或零
            
            metadata['reason'] = 'high_factor_score'
            
            # 如果需要再平衡，更新投资组合
            if need_rebalance:
                self._update_portfolio('LONG', composite_score, expected_excess_return)
        
        # 空头信号（低分股票，后10%）
        elif composite_score < self.bottom_percentile:
            signal_strength = (self.bottom_percentile - composite_score) / self.bottom_percentile
            signal_strength = min(signal_strength, 1.0)
            direction = 'SHORT'
            
            # 置信度基于超额收益和行业中性程度
            excess_confidence = min(abs(expected_excess_return) / 0.05, 1.0)
            neutral_confidence = industry_neutrality
            confidence = signal_strength * (excess_confidence * 0.6 + neutral_confidence * 0.4)
            confidence = max(confidence, 0.1)  # 确保置信度不为负数或零
            
            metadata['reason'] = 'low_factor_score'
            
            # 如果需要再平衡，更新投资组合
            if need_rebalance:
                self._update_portfolio('SHORT', composite_score, expected_excess_return)
        
        # 中性区域
        else:
            signal_strength = 0.3
            direction = 'NEUTRAL'
            confidence = 0.4
            metadata['reason'] = 'neutral_score'
        
        # 如果需要再平衡，更新时间戳
        if need_rebalance:
            self.last_rebalance_time = current_time
        
        signal = {
            'signal_strength': signal_strength,
            'direction': direction,
            'confidence': confidence,
            'suggested_positions': {},
            'metadata': metadata
        }
        
        self.record_signal(signal)
        return signal
    
    def _update_portfolio(self, direction: str, score: float, expected_return: float):
        """
        更新投资组合（文档要求：定期再平衡）
        
        Args:
            direction: 方向（LONG/SHORT）
            score: 因子得分
            expected_return: 预期超额收益
        """
        # 简化实现：记录投资组合信息
        portfolio_id = f"{direction}_{len(self.current_portfolio)}"
        self.current_portfolio[portfolio_id] = {
            'direction': direction,
            'score': score,
            'expected_return': expected_return,
            'entry_time': self.last_rebalance_time
        }
        
        # 保持投资组合规模（简化：最多10个持仓）
        if len(self.current_portfolio) > 10:
            # 移除预期收益最低的持仓
            worst_id = min(self.current_portfolio.keys(), 
                          key=lambda k: self.current_portfolio[k]['expected_return'])
            del self.current_portfolio[worst_id]
    
    def _calculate_simple_factor_score(self, indicators: Dict[str, Any]) -> float:
        """
        计算简单的多因子得分
        
        使用技术指标构建简单的因子得分：
        - 动量因子：RSI
        - 趋势因子：MACD
        - 波动率因子：ATR
        """
        # RSI因子（归一化到0-1）
        rsi = indicators.get('RSI', 50)
        rsi_score = rsi / 100
        
        # MACD因子
        macd_hist = indicators.get('MACD_HIST', 0)
        macd_score = 0.5 + np.tanh(macd_hist) * 0.5  # 归一化到0-1
        
        # 趋势因子（MA5 vs MA20）
        ma5 = indicators.get('MA5', 0)
        ma20 = indicators.get('MA20', 0)
        if ma20 > 0:
            trend_score = 0.5 + np.tanh((ma5 - ma20) / ma20) * 0.5
        else:
            trend_score = 0.5
        
        # 综合得分（加权平均）
        factor_score = 0.4 * rsi_score + 0.3 * macd_score + 0.3 * trend_score
        
        return factor_score
    
    def get_required_indicators(self) -> List[str]:
        """返回所需指标（文档要求的多因子模型指标）"""
        return [
            'COMPOSITE_SCORE',  # 多因子综合得分
            'EXCESS_RETURN',    # 超额收益
            'INDUSTRY_EXPOSURE', # 行业暴露度
            'VOLATILITY',       # 波动率
            'VALUE_FACTOR',     # 价值因子
            'MOMENTUM_FACTOR',  # 动量因子
            'QUALITY_FACTOR'    # 质量因子
        ]
