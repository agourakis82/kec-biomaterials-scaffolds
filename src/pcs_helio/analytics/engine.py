"""
PCS Helio Analytics Engine
=========================

Motor avançado de analytics para processamento de dados complexos
e integração com pipeline KEC.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsConfig:
    """Configuração do motor de analytics."""
    enable_advanced_stats: bool = True
    enable_ml_insights: bool = True
    parallel_processing: bool = True
    cache_results: bool = True
    max_workers: int = 4
    analysis_timeout: float = 30.0


@dataclass
class AnalysisResults:
    """Resultados de análise completa."""
    summary_stats: Dict[str, float]
    trends: Dict[str, List[float]]
    correlations: Dict[str, float]
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AnalyticsEngine:
    """
    Motor de analytics avançados com funcionalidades:
    - Análise estatística multivariada
    - Detecção de padrões temporais
    - Correlações cruzadas
    - Insights automatizados
    - Integração com ML pipelines
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        self._cache: Dict[str, Any] = {}
        
    async def analyze_kec_metrics(self, 
                                metrics_data: Union[Dict[str, float], List[Dict[str, float]]], 
                                metadata: Optional[Dict[str, Any]] = None) -> AnalysisResults:
        """
        Análise avançada de métricas KEC.
        
        Args:
            metrics_data: Métricas KEC (single ou time series)
            metadata: Metadados adicionais
            
        Returns:
            AnalysisResults com insights
        """
        
        logger.info("Iniciando análise avançada de métricas KEC")
        
        # Normaliza input para lista
        if isinstance(metrics_data, dict):
            data_list = [metrics_data]
        else:
            data_list = metrics_data
            
        # Converte para DataFrame para análise
        df = pd.DataFrame(data_list)
        
        # Análises estatísticas
        summary_stats = await self._compute_summary_statistics(df)
        
        # Análise de tendências (se múltiplos pontos)
        trends = await self._analyze_trends(df) if len(df) > 1 else {}
        
        # Correlações
        correlations = await self._compute_correlations(df)
        
        # Insights automatizados
        insights = await self._generate_insights(df, summary_stats, correlations)
        
        return AnalysisResults(
            summary_stats=summary_stats,
            trends=trends,
            correlations=correlations,
            insights=insights,
            metadata=metadata or {}
        )
    
    async def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Computa estatísticas resumidas."""
        
        stats = {}
        
        # Estatísticas básicas para métricas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[f"{col}_mean"] = float(df[col].mean())
            stats[f"{col}_std"] = float(df[col].std())
            stats[f"{col}_min"] = float(df[col].min())
            stats[f"{col}_max"] = float(df[col].max())
            
            # Quartis
            stats[f"{col}_q25"] = float(df[col].quantile(0.25))
            stats[f"{col}_q50"] = float(df[col].quantile(0.50))
            stats[f"{col}_q75"] = float(df[col].quantile(0.75))
        
        # Estatísticas globais
        if len(df) > 1:
            stats["total_samples"] = len(df)
            stats["missing_data_pct"] = float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        
        return stats
    
    async def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Analisa tendências temporais."""
        
        trends = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = df[col].dropna().tolist()
            if len(values) > 2:
                # Trend simples: diferenças consecutivas
                diffs = np.diff(values)
                trends[f"{col}_trend"] = diffs.tolist()
                
                # Slope linear (regressão simples)
                x = np.arange(len(values))
                slope = float(np.polyfit(x, values, 1)[0])
                trends[f"{col}_slope"] = [slope]
        
        return trends
    
    async def _compute_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Computa matriz de correlações."""
        
        correlations = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Extrai correlações significativas (upper triangle)
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr_value = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value):
                        correlations[f"{col1}_vs_{col2}"] = float(corr_value)
        
        return correlations
    
    async def _generate_insights(self, 
                               df: pd.DataFrame, 
                               stats: Dict[str, float], 
                               correlations: Dict[str, float]) -> List[str]:
        """Gera insights automatizados."""
        
        insights = []
        
        # Insight sobre distribuição de métricas KEC
        if "H_spectral_mean" in stats:
            h_mean = stats["H_spectral_mean"]
            if h_mean > 3.0:
                insights.append("Alta entropia espectral detectada - estrutura complexa")
            elif h_mean < 1.0:
                insights.append("Baixa entropia espectral - estrutura mais ordenada")
        
        if "sigma_mean" in stats:
            sigma_mean = stats["sigma_mean"]
            if sigma_mean > 2.0:
                insights.append("Forte comportamento small-world detectado")
            elif sigma_mean < 1.2:
                insights.append("Comportamento small-world fraco ou ausente")
        
        # Insights sobre correlações
        strong_correlations = {k: v for k, v in correlations.items() if abs(v) > 0.7}
        if strong_correlations:
            for corr_name, corr_value in strong_correlations.items():
                direction = "positiva" if corr_value > 0 else "negativa"
                insights.append(f"Correlação {direction} forte: {corr_name} ({corr_value:.3f})")
        
        # Insight sobre variabilidade
        if len(df) > 1:
            high_variance_metrics = []
            for col in df.select_dtypes(include=[np.number]).columns:
                cv = df[col].std() / (df[col].mean() + 1e-9)  # Coefficient of variation
                if cv > 0.3:  # Alta variabilidade
                    high_variance_metrics.append(col)
            
            if high_variance_metrics:
                insights.append(f"Alta variabilidade detectada em: {', '.join(high_variance_metrics)}")
        
        # Insight sobre outliers (IQR method)
        for col in df.select_dtypes(include=[np.number]).columns:
            if len(df) > 4:  # Mínimo para IQR
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                if len(outliers) > 0:
                    insights.append(f"Outliers detectados em {col}: {len(outliers)} amostras")
        
        return insights
    
    async def comparative_analysis(self, 
                                 datasets: Dict[str, Union[Dict[str, float], List[Dict[str, float]]]]) -> Dict[str, AnalysisResults]:
        """Análise comparativa entre múltiplos datasets."""
        
        results = {}
        
        for name, data in datasets.items():
            logger.info(f"Analisando dataset: {name}")
            results[name] = await self.analyze_kec_metrics(data, {"dataset_name": name})
        
        # Adiciona insights comparativos
        if len(results) > 1:
            comparative_insights = self._generate_comparative_insights(results)
            for name, result in results.items():
                result.insights.extend(comparative_insights.get(name, []))
        
        return results
    
    def _generate_comparative_insights(self, results: Dict[str, AnalysisResults]) -> Dict[str, List[str]]:
        """Gera insights comparativos entre datasets."""
        
        insights = {name: [] for name in results.keys()}
        
        # Compara médias de métricas principais
        for metric in ["H_spectral", "sigma", "k_forman_mean"]:
            means = {}
            for name, result in results.items():
                mean_key = f"{metric}_mean"
                if mean_key in result.summary_stats:
                    means[name] = result.summary_stats[mean_key]
            
            if len(means) > 1:
                sorted_datasets = sorted(means.items(), key=lambda x: x[1], reverse=True)
                highest = sorted_datasets[0]
                lowest = sorted_datasets[-1]
                
                if len(sorted_datasets) > 1:
                    insights[highest[0]].append(f"Maior {metric} comparativamente ({highest[1]:.3f})")
                    insights[lowest[0]].append(f"Menor {metric} comparativamente ({lowest[1]:.3f})")
        
        return insights
    
    async def get_health_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de saúde do analytics engine."""
        
        return {
            "engine": "pcs_helio_analytics",
            "cache_size": len(self._cache),
            "config": self.config.__dict__,
            "status": "ready"
        }