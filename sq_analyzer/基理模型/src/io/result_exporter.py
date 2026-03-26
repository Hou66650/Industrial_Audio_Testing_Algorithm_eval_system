"""结果导出模块 - 支持原始值和量化分数"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from loguru import logger


class ResultExporter:
    """结果导出器 - 支持原始值和0-100量化分数"""
    
    @staticmethod
    def to_csv(
        results: Dict[str, Dict],
        output_path: Path,
        include_scores: bool = True
    ) -> Path:
        """
        导出为CSV格式 - 宽表格式
        
        Args:
            results: 特征结果字典（已包含分数）
            output_path: 输出路径
            include_scores: 是否包含分数字段
        """
        import pandas as pd
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建数据行
        rows = []
        for sample_name, features in results.items():
            row = {
                'sample_name': sample_name,
                'label': ResultExporter._get_label_from_filename(sample_name)
            }
            
            # 提取每个特征的值和分数
            for feature_name, data in features.items():
                if isinstance(data, dict) and 'value' in data:
                    # 原始值
                    row[f"{feature_name}_raw"] = data.get('value')
                    row[f"{feature_name}_unit"] = data.get('unit', '')
                    
                    # 分数
                    if include_scores:
                        row[f"{feature_name}_adaptive"] = data.get('adaptive_score')
                        row[f"{feature_name}_fixed"] = data.get('fixed_score')
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"CSV导出完成: {output_path}")
        return output_path
    
    @staticmethod
    def to_excel(
        results: Dict[str, Dict],
        output_path: Path,
        normalizer: Optional[Any] = None
    ) -> Path:
        """
        导出为Excel格式 - 包含多个sheet
        
        Sheets:
        1. 原始数据 (raw_data): 原始计算值
        2. 自适应分数 (adaptive_scores): 基于数据自适应归一化的0-100分数
        3. 固定分数 (fixed_scores): 基于固定范围的0-100分数
        4. 范围信息 (ranges): 归一化使用的范围定义
        5. 统计摘要 (statistics): OK/NG的统计对比
        
        Args:
            results: 特征结果字典（已包含分数）
            output_path: 输出路径
            normalizer: 归一化器实例（用于输出范围信息）
        """
        import pandas as pd
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 获取特征列表
            feature_cols = []
            for sample_data in results.values():
                for key, val in sample_data.items():
                    if isinstance(val, dict) and 'value' in val:
                        feature_cols.append(key)
                break
            
            # Sheet 1: 原始数据
            raw_rows = []
            for sample_name, features in results.items():
                row = {
                    'sample_name': sample_name,
                    'label': ResultExporter._get_label_from_filename(sample_name)
                }
                for feat in feature_cols:
                    if feat in features and isinstance(features[feat], dict):
                        row[feat] = features[feat].get('value')
                raw_rows.append(row)
            
            df_raw = pd.DataFrame(raw_rows)
            df_raw.to_excel(writer, sheet_name='原始数据', index=False)
            
            # Sheet 2: 自适应分数
            adaptive_rows = []
            for sample_name, features in results.items():
                row = {
                    'sample_name': sample_name,
                    'label': ResultExporter._get_label_from_filename(sample_name)
                }
                for feat in feature_cols:
                    if feat in features and isinstance(features[feat], dict):
                        row[feat] = features[feat].get('adaptive_score')
                adaptive_rows.append(row)
            
            df_adaptive = pd.DataFrame(adaptive_rows)
            df_adaptive.to_excel(writer, sheet_name='自适应分数', index=False)
            
            # Sheet 3: 固定分数
            fixed_rows = []
            for sample_name, features in results.items():
                row = {
                    'sample_name': sample_name,
                    'label': ResultExporter._get_label_from_filename(sample_name)
                }
                for feat in feature_cols:
                    if feat in features and isinstance(features[feat], dict):
                        row[feat] = features[feat].get('fixed_score')
                fixed_rows.append(row)
            
            df_fixed = pd.DataFrame(fixed_rows)
            df_fixed.to_excel(writer, sheet_name='固定分数', index=False)
            
            # Sheet 4: 范围信息
            if normalizer:
                range_rows = []
                for feat in feature_cols:
                    info = normalizer.get_range_info(feat)
                    row = {'feature': feat}
                    
                    if 'adaptive' in info:
                        row['adaptive_min'] = info['adaptive']['min']
                        row['adaptive_max'] = info['adaptive']['max']
                    
                    if 'fixed' in info:
                        row['fixed_min'] = info['fixed']['min']
                        row['fixed_max'] = info['fixed']['max']
                    
                    range_rows.append(row)
                
                df_range = pd.DataFrame(range_rows)
                df_range.to_excel(writer, sheet_name='范围定义', index=False)
            
            # Sheet 5: 统计摘要
            stats = ResultExporter._calculate_statistics(results, feature_cols)
            df_stats = pd.DataFrame(stats)
            df_stats.to_excel(writer, sheet_name='统计摘要', index=False)
        
        logger.info(f"Excel导出完成: {output_path}")
        return output_path
    
    @staticmethod
    def to_json(
        results: Dict[str, Dict],
        output_path: Path,
        indent: int = 2
    ) -> Path:
        """
        导出为JSON格式 - 完整结构化数据
        
        Args:
            results: 特征结果字典（已包含分数）
            output_path: 输出路径
            indent: 缩进空格数
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建输出结构
        output = {
            'metadata': {
                'total_samples': len(results),
                'ok_count': sum(1 for k in results if ResultExporter._get_label_from_filename(k) == 'OK'),
                'ng_count': sum(1 for k in results if ResultExporter._get_label_from_filename(k) == 'NG'),
                'feature_count': 0
            },
            'samples': {}
        }
        
        # 处理样本
        for sample_name, features in results.items():
            sample_data = {
                'label': ResultExporter._get_label_from_filename(sample_name),
                'features': {}
            }
            
            for feature_name, data in features.items():
                if isinstance(data, dict) and 'value' in data:
                    sample_data['features'][feature_name] = {
                        'raw': ResultExporter._convert_to_native(data.get('value')),
                        'unit': data.get('unit', ''),
                        'adaptive_score': ResultExporter._convert_to_native(data.get('adaptive_score')),
                        'fixed_score': ResultExporter._convert_to_native(data.get('fixed_score'))
                    }
            
            output['samples'][sample_name] = sample_data
            
            if output['metadata']['feature_count'] == 0:
                output['metadata']['feature_count'] = len(sample_data['features'])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=indent)
        
        logger.info(f"JSON导出完成: {output_path}")
        return output_path
    
    @staticmethod
    def _get_label_from_filename(filename: str) -> str:
        """从文件名获取标签"""
        name_lower = filename.lower()
        if 'ng' in name_lower:
            return 'NG'
        elif 'ok' in name_lower:
            return 'OK'
        return 'UNKNOWN'
    
    @staticmethod
    def _convert_to_native(obj):
        """转换numpy类型为Python原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj) if not (np.isnan(obj) or np.isinf(obj)) else None
        return obj
    
    @staticmethod
    def _calculate_statistics(results: Dict, feature_cols: List[str]) -> List[Dict]:
        """计算统计摘要"""
        stats = []
        
        # 分离OK和NG
        ok_samples = {k: v for k, v in results.items() 
                     if ResultExporter._get_label_from_filename(k) == 'OK'}
        ng_samples = {k: v for k, v in results.items() 
                     if ResultExporter._get_label_from_filename(k) == 'NG'}
        
        for feat in feature_cols:
            stat = {'feature': feat}
            
            # 原始值统计
            ok_raw = [s[feat]['value'] for s in ok_samples.values() 
                     if feat in s and isinstance(s[feat], dict) and s[feat].get('value') is not None]
            ng_raw = [s[feat]['value'] for s in ng_samples.values() 
                     if feat in s and isinstance(s[feat], dict) and s[feat].get('value') is not None]
            
            if ok_raw:
                stat['ok_raw_mean'] = np.mean(ok_raw)
                stat['ok_raw_std'] = np.std(ok_raw)
            
            if ng_raw:
                stat['ng_raw_mean'] = np.mean(ng_raw)
                stat['ng_raw_std'] = np.std(ng_raw)
            
            # 自适应分数统计
            ok_adaptive = [s[feat]['adaptive_score'] for s in ok_samples.values() 
                          if feat in s and isinstance(s[feat], dict) and s[feat].get('adaptive_score') is not None]
            ng_adaptive = [s[feat]['adaptive_score'] for s in ng_samples.values() 
                          if feat in s and isinstance(s[feat], dict) and s[feat].get('adaptive_score') is not None]
            
            if ok_adaptive:
                stat['ok_adaptive_mean'] = np.mean(ok_adaptive)
            if ng_adaptive:
                stat['ng_adaptive_mean'] = np.mean(ng_adaptive)
            
            # 固定分数统计
            ok_fixed = [s[feat]['fixed_score'] for s in ok_samples.values() 
                       if feat in s and isinstance(s[feat], dict) and s[feat].get('fixed_score') is not None]
            ng_fixed = [s[feat]['fixed_score'] for s in ng_samples.values() 
                       if feat in s and isinstance(s[feat], dict) and s[feat].get('fixed_score') is not None]
            
            if ok_fixed:
                stat['ok_fixed_mean'] = np.mean(ok_fixed)
            if ng_fixed:
                stat['ng_fixed_mean'] = np.mean(ng_fixed)
            
            # Cohen's d (基于原始值)
            if ok_raw and ng_raw and stat.get('ok_raw_std', 0) > 0:
                stat['cohens_d'] = (stat['ng_raw_mean'] - stat['ok_raw_mean']) / stat['ok_raw_std']
            
            stats.append(stat)
        
        return stats
