"""
资金分配器 (Position Allocator)

根据策略信号分配资金到不同股票
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class PositionAllocator:
    """
    资金分配器
    
    根据策略信号强度和置信度分配资金
    """
    
    def __init__(self, 
                 total_capital: float = 1000000,
                 max_position_ratio: float = 0.3,      # 单股票最大仓位比例
                 total_position_ratio: float = 0.8,    # 总仓位上限比例
                 allow_short: bool = True,
                 min_signal_threshold: float = 0.08,
                 signal_power: float = 1.5):
        """
        初始化资金分配器
        
        Args:
            total_capital: 总资金
            max_position_ratio: 单股票最大仓位比例（如0.3表示30%）
            total_position_ratio: 总仓位上限比例（如0.8表示80%）
        """
        self.total_capital = total_capital
        self.max_position_ratio = max_position_ratio
        self.total_position_ratio = total_position_ratio
        self.allow_short = allow_short
        self.min_signal_threshold = max(float(min_signal_threshold), 0.0)
        self.signal_power = max(float(signal_power), 1.0)
    
    def allocate_by_signal_strength(self, 
                                    signals: Dict[str, Dict[str, Any]], 
                                    selected_strategy: int) -> Dict[str, float]:
        """
        根据信号强度分配资金（修复版：正确处理LONG/SHORT信号）
        
        Args:
            signals: 所有股票的信号字典
                {
                    symbol: {
                        'signal_strength': float,
                        'confidence': float,
                        'direction': str,  # 'LONG', 'SHORT', 'NEUTRAL'
                        ...
                    }
                }
            selected_strategy: 选中的策略ID
        
        Returns:
            positions: 资金分配结果 {symbol: capital_amount}
                      正值表示LONG仓位，负值表示SHORT仓位
        """
        positions = {}

        # 将动作0作为稳态基线动作：等权长仓（类Buy & Hold）
        # 目的：给RL提供稳定正向锚点，避免在弱信号期退化为纯噪声切换。
        if selected_strategy == 0 and signals:
            baseline_symbols = sorted(list(signals.keys()))
            return self.allocate_equal_weight(
                symbols=baseline_symbols,
                num_positions=len(baseline_symbols)
            )
        
        # 收集所有股票的加权信号强度（区分方向）
        weighted_strengths = {}
        long_aliases = {'LONG', 'BUY_SPREAD', 'LONG_A_SHORT_B'}
        short_aliases = {'SHORT', 'SELL_SPREAD', 'SHORT_A_LONG_B'}

        for symbol, signal in signals.items():
            strength = signal.get('signal_strength', 0)
            confidence = signal.get('confidence', 0)
            direction = signal.get('direction', 'NEUTRAL')
            
            # 综合信号强度和置信度
            weighted_strength = strength * confidence
            if abs(weighted_strength) < self.min_signal_threshold:
                continue
            
            # 兼容多种方向语义，统一映射到LONG/SHORT
            if direction in long_aliases and weighted_strength > 0:
                weighted_strengths[symbol] = weighted_strength
            elif direction in short_aliases and weighted_strength > 0:
                weighted_strengths[symbol] = -weighted_strength  # 负值表示SHORT
        
        # 如果没有有效信号，返回空仓位
        if not weighted_strengths:
            return positions
        
        # 分别处理LONG和SHORT信号（重新启用SHORT）
        long_strengths = {k: v for k, v in weighted_strengths.items() if v > 0}
        short_strengths = {k: abs(v) for k, v in weighted_strengths.items() if v < 0}  # 重新启用SHORT，取绝对值
        
        # 归一化LONG信号
        if long_strengths:
            long_strengths = {k: abs(v) ** self.signal_power for k, v in long_strengths.items()}
            total_long = sum(long_strengths.values())
            if total_long > 0:
                for symbol in long_strengths:
                    long_strengths[symbol] /= total_long
        
        # 归一化SHORT信号
        if short_strengths:
            short_strengths = {k: abs(v) ** self.signal_power for k, v in short_strengths.items()}
            total_short = sum(short_strengths.values())
            if total_short > 0:
                for symbol in short_strengths:
                    short_strengths[symbol] /= total_short
        
        # 总目标仓位由配置控制，避免硬编码压低资金利用率
        total_ratio = min(max(self.total_position_ratio, 0.0), 1.0)
        available_capital = self.total_capital * total_ratio
        
        # 分配LONG仓位（合理资金使用）
        total_allocated = 0
        for symbol, weight in long_strengths.items():
            allocated_capital = available_capital * weight
            max_capital = self.total_capital * min(max(self.max_position_ratio, 0.0), 1.0)
            allocated_capital = min(allocated_capital, max_capital)
            
            # 检查总分配是否超出可用资金
            if total_allocated + allocated_capital <= available_capital:
                positions[symbol] = allocated_capital  # 正值表示LONG
                total_allocated += allocated_capital
            else:
                # 分配剩余资金
                remaining = available_capital - total_allocated
                if remaining > 0:
                    positions[symbol] = remaining
                    total_allocated = available_capital
                break
        
        # 分配SHORT仓位（按开关控制）
        if self.allow_short:
            for symbol, weight in short_strengths.items():
                allocated_capital = available_capital * weight * 0.15  # 做空权重更保守
                max_capital = self.total_capital * min(max(self.max_position_ratio * 0.5, 0.0), 0.15)
                allocated_capital = min(allocated_capital, max_capital)
                
                # 检查总分配是否超出可用资金
                if total_allocated + allocated_capital <= available_capital:
                    positions[symbol] = -allocated_capital  # 负值表示SHORT
                    total_allocated += allocated_capital
                else:
                    # 分配剩余资金
                    remaining = available_capital - total_allocated
                    if remaining > 0:
                        positions[symbol] = -remaining
                        total_allocated = available_capital
                    break
        
        return positions
    
    def allocate_equal_weight(self, 
                             symbols: list, 
                             num_positions: int = None) -> Dict[str, float]:
        """
        等权分配资金
        
        Args:
            symbols: 股票代码列表
            num_positions: 持仓数量（如果为None，则对所有股票等权）
        
        Returns:
            positions: 资金分配结果
        """
        if not symbols:
            return {}
        
        if num_positions is None:
            num_positions = len(symbols)
        else:
            num_positions = min(num_positions, len(symbols))
        
        # 计算每个股票的资金
        available_capital = self.total_capital * min(max(self.total_position_ratio, 0.0), 1.0)
        capital_per_stock = available_capital / num_positions
        
        # 应用单股票上限
        max_capital = self.total_capital * min(max(self.max_position_ratio, 0.0), 1.0)
        capital_per_stock = min(capital_per_stock, max_capital)
        
        positions = {symbol: capital_per_stock for symbol in symbols[:num_positions]}
        
        return positions
    
    def update_total_capital(self, new_capital: float):
        """更新总资金（增加异常值保护）"""
        # 检查新资金是否合理（静默处理异常值）
        if (pd.isna(new_capital) or np.isinf(new_capital) or 
            new_capital <= 0 or new_capital > 1000000 * 100):  # 不应超过初始资金的100倍
            # 静默保持原值，不输出警告
            return
        
        self.total_capital = new_capital
    
    def get_max_position_value(self) -> float:
        """获取单股票最大仓位金额"""
        return self.total_capital * self.max_position_ratio
    
    def get_total_position_limit(self) -> float:
        """获取总仓位上限金额"""
        return self.total_capital * self.total_position_ratio
    
    def __str__(self) -> str:
        return (f"PositionAllocator(capital={self.total_capital:.2f}, "
                f"max_position={self.max_position_ratio:.1%}, "
                f"total_limit={self.total_position_ratio:.1%})")
    
    def __repr__(self) -> str:
        return self.__str__()
