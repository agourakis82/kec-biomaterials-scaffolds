"""Mathematical Contracts - Contratos Matemáticos Especializados

Implementações matemáticas avançadas para análise de biomateriais, redes e sistemas complexos.
Integração com NetworkX, NumPy, SciPy para cálculos KEC e métricas topológicas.

Feature crítica #5 para mestrado - Contratos matemáticos especializados.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import json

logger = logging.getLogger(__name__)


# ============================================================================
# CORE MATHEMATICAL FUNCTIONS
# ============================================================================

def safe_log(x: float, base: float = math.e) -> float:
    """Logaritmo seguro que evita log(0)."""
    return math.log(max(x, 1e-10), base)


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normaliza score para intervalo especificado."""
    return max(min_val, min(max_val, score))


def calculate_entropy(probabilities: List[float]) -> float:
    """Calcula entropia de Shannon de uma distribuição."""
    if not probabilities:
        return 0.0
    
    # Normaliza probabilidades
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    
    probs = [p / total for p in probabilities if p > 0]
    return -sum(p * safe_log(p, 2) for p in probs)


def calculate_mutual_information(joint_probs: List[List[float]]) -> float:
    """Calcula informação mútua de distribuição conjunta."""
    if not joint_probs or not joint_probs[0]:
        return 0.0
    
    # Marginals
    marginal_x = [sum(row) for row in joint_probs]
    marginal_y = [sum(joint_probs[i][j] for i in range(len(joint_probs))) 
                  for j in range(len(joint_probs[0]))]
    
    # Mutual information
    mi = 0.0
    for i, row in enumerate(joint_probs):
        for j, p_xy in enumerate(row):
            if p_xy > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                mi += p_xy * safe_log(p_xy / (marginal_x[i] * marginal_y[j]), 2)
    
    return mi


# ============================================================================
# DELTA KEC CONTRACT
# ============================================================================

def delta_kec_v1_contract(input_data: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Contrato Delta KEC v1 - Core do mestrado.
    
    Calcula o coeficiente de transferência de conhecimento Delta KEC
    considerando entropias de fonte/alvo e informação mútua.
    
    Args:
        input_data: Dados com source_entropy, target_entropy, mutual_information
        parameters: Parâmetros alpha, beta para ponderação
        
    Returns:
        Dict com score, confidence, delta_kec, e metadados
    """
    
    # Extrai dados
    source_entropy = float(input_data.get('source_entropy', 0))
    target_entropy = float(input_data.get('target_entropy', 0))
    mutual_info = float(input_data.get('mutual_information', 0))
    
    # Parâmetros de ponderação
    alpha = parameters.get('alpha', 0.7) if parameters else 0.7
    beta = parameters.get('beta', 0.3) if parameters else 0.3
    
    # Validação básica
    if source_entropy < 0 or target_entropy < 0:
        return {
            'error': 'Entropy values must be non-negative',
            'score': 0.0,
            'confidence': 0.0
        }
    
    # Calcula Delta KEC
    delta_kec = target_entropy - source_entropy
    
    # Normalização considerando informação mútua
    if mutual_info > 0:
        # Informação mútua reduz a incerteza efetiva
        effective_source = source_entropy - mutual_info
        effective_target = target_entropy - mutual_info
        normalized_delta = effective_target - effective_source
    else:
        normalized_delta = delta_kec
    
    # Score final ponderado
    # Positivo indica transferência eficiente (target mais organizado)
    # Negativo indica perda de informação
    raw_score = alpha * normalized_delta + beta * mutual_info
    
    # Normaliza score para [-1, 1]
    max_possible_entropy = max(source_entropy, target_entropy, 10.0)
    normalized_score = raw_score / max_possible_entropy if max_possible_entropy > 0 else 0
    normalized_score = max(-1.0, min(1.0, normalized_score))
    
    # Confidence baseada na magnitude e consistência dos valores
    total_entropy = source_entropy + target_entropy
    entropy_balance = 1.0 - abs(source_entropy - target_entropy) / (total_entropy + 1e-10)
    
    if total_entropy > 0:
        confidence = min(0.95, (entropy_balance * 0.7 + min(1.0, total_entropy / 20.0) * 0.3))
    else:
        confidence = 0.1
    
    # Score final para resposta (convertido para [0, 1])
    final_score = (normalized_score + 1.0) / 2.0
    
    # Classificação qualitativa
    if normalized_score > 0.5:
        transfer_quality = "excellent"
    elif normalized_score > 0.1:
        transfer_quality = "good"
    elif normalized_score > -0.1:
        transfer_quality = "moderate"
    elif normalized_score > -0.5:
        transfer_quality = "poor"
    else:
        transfer_quality = "very_poor"
    
    return {
        'score': float(final_score),
        'confidence': float(confidence),
        'delta_kec': float(delta_kec),
        'normalized_delta': float(normalized_delta),
        'raw_score': float(raw_score),
        'transfer_quality': transfer_quality,
        'metadata': {
            'source_entropy': source_entropy,
            'target_entropy': target_entropy,
            'mutual_information': mutual_info,
            'alpha': alpha,
            'beta': beta,
            'entropy_balance': float(entropy_balance),
            'max_possible_entropy': float(max_possible_entropy)
        }
    }


# ============================================================================
# ZUCO READING CONTRACT
# ============================================================================

def zuco_reading_v1_contract(input_data: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Contrato ZuCo Reading v1 - EEG + Eye Tracking.
    
    Analisa compreensão de leitura baseada em features de EEG e eye tracking
    do corpus ZuCo (Zurich Cognitive Load Corpus).
    
    Args:
        input_data: Features EEG e eye tracking
        parameters: Pesos para EEG e eye tracking
        
    Returns:
        Dict com score, confidence, difficulty e metadados
    """
    
    # Extrai features
    eeg = input_data.get('eeg_features', {})
    et = input_data.get('eye_tracking_features', {})
    
    # Valida features obrigatórias
    required_eeg = ['theta_power', 'alpha_power', 'beta_power', 'gamma_power']
    if not all(feature in eeg for feature in required_eeg):
        return {
            'error': f'Missing required EEG features: {required_eeg}',
            'score': 0.0,
            'confidence': 0.0
        }
    
    # Pesos para combinação
    eeg_weight = parameters.get('eeg_weight', 0.6) if parameters else 0.6
    et_weight = parameters.get('et_weight', 0.4) if parameters else 0.4
    
    # === ANÁLISE EEG ===
    
    # Poder total das bandas
    eeg_total = sum(eeg[band] for band in required_eeg)
    
    if eeg_total <= 0:
        return {
            'error': 'EEG power values must be positive',
            'score': 0.0,
            'confidence': 0.0
        }
    
    # Distribuição relativa das bandas
    eeg_dist = {band: eeg[band] / eeg_total for band in required_eeg}
    
    # Score EEG baseado em literatura cognitiva
    # Alpha alta = relaxamento, melhor compreensão
    # Theta moderado = atenção focada
    # Beta controlado = processamento ativo
    # Gamma baixo = menos sobrecarga cognitiva
    
    alpha_score = min(1.0, eeg_dist['alpha_power'] * 4)  # Ideal ~25%
    theta_score = min(1.0, eeg_dist['theta_power'] * 5)  # Ideal ~20%
    beta_score = 1.0 - min(1.0, eeg_dist['beta_power'] * 3)  # Penaliza excesso
    gamma_score = 1.0 - min(1.0, eeg_dist['gamma_power'] * 6)  # Penaliza excesso
    
    eeg_balance_score = (alpha_score * 0.4 + theta_score * 0.3 + 
                        beta_score * 0.2 + gamma_score * 0.1)
    
    # === ANÁLISE EYE TRACKING ===
    
    et_score = 0.5  # Default se não houver features
    
    if et:
        # Fixation duration (maior = melhor compreensão)
        fix_duration = et.get('avg_fixation_duration', 200)
        fix_score = min(1.0, fix_duration / 300.0)  # 300ms = baseline
        
        # Saccade velocity (menor = mais deliberado)
        saccade_vel = et.get('avg_saccade_velocity', 180)
        saccade_score = max(0.0, 1.0 - (saccade_vel - 150) / 200)  # Ideal 150-200°/s
        
        # Regression rate (menor = melhor)
        regression_rate = et.get('regression_rate', 0.15)
        regression_score = max(0.0, 1.0 - regression_rate * 2)  # Penaliza regressões
        
        # Reading speed (moderada é melhor)
        reading_speed = et.get('reading_speed_wpm', 200)
        speed_score = 1.0 - abs(reading_speed - 250) / 250  # Ideal ~250 WPM
        speed_score = max(0.0, speed_score)
        
        et_score = (fix_score * 0.3 + saccade_score * 0.2 + 
                   regression_score * 0.3 + speed_score * 0.2)
    
    # === SCORE FINAL ===
    
    combined_score = eeg_weight * eeg_balance_score + et_weight * et_score
    
    # Cognitive load estimation
    cognitive_load = eeg_dist['gamma_power'] + eeg_dist['beta_power'] / 2
    cognitive_load = min(1.0, cognitive_load * 2)  # Normaliza
    
    # Reading difficulty classification
    if combined_score > 0.8:
        difficulty = "easy"
        difficulty_level = 1
    elif combined_score > 0.65:
        difficulty = "moderate"
        difficulty_level = 2
    elif combined_score > 0.4:
        difficulty = "hard"
        difficulty_level = 3
    else:
        difficulty = "very_hard"
        difficulty_level = 4
    
    # Confidence baseada na consistência das medidas
    eeg_consistency = 1.0 - abs(alpha_score - theta_score)
    et_consistency = (fix_score + (1-regression_score)) / 2 if et else 0.5
    overall_consistency = eeg_weight * eeg_consistency + et_weight * et_consistency
    
    confidence = min(0.95, combined_score * 0.6 + overall_consistency * 0.4)
    
    return {
        'score': float(combined_score),
        'confidence': float(confidence),
        'reading_difficulty': difficulty,
        'difficulty_level': difficulty_level,
        'cognitive_load': float(cognitive_load),
        'metadata': {
            'eeg_score': float(eeg_balance_score),
            'et_score': float(et_score),
            'eeg_distribution': eeg_dist,
            'eeg_consistency': float(eeg_consistency),
            'et_consistency': float(et_consistency),
            'weights_used': {'eeg': eeg_weight, 'et': et_weight}
        }
    }


# ============================================================================
# EDITORIAL CONTRACT
# ============================================================================

def editorial_v1_contract(input_data: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Contrato Editorial v1 - Análise de qualidade textual.
    
    Avalia qualidade editorial usando múltiplas dimensões:
    readability, grammar, vocabulary, coherence, originality.
    
    Args:
        input_data: Métricas textuais
        parameters: Pesos para cada dimensão
        
    Returns:
        Dict com score, categoria de qualidade e recomendações
    """
    
    metrics = input_data.get('text_metrics', {})
    if not metrics:
        return {
            'error': 'Missing text_metrics in input data',
            'score': 0.0,
            'confidence': 0.0
        }
    
    # Pesos padrão baseados em literatura de escrita acadêmica
    default_weights = {
        'readability': 0.2,
        'grammar': 0.25,
        'vocabulary': 0.15,
        'coherence': 0.25,
        'originality': 0.15
    }
    
    weights = parameters.get('weights', default_weights) if parameters else default_weights
    
    # Normaliza métricas (assume escala 0-100 exceto vocabulary_diversity)
    normalized_metrics = {}
    
    # Readability (0-100)
    readability = metrics.get('readability_score', 50) / 100.0
    normalized_metrics['readability'] = max(0.0, min(1.0, readability))
    
    # Grammar (0-100)
    grammar = metrics.get('grammar_score', 50) / 100.0
    normalized_metrics['grammar'] = max(0.0, min(1.0, grammar))
    
    # Vocabulary diversity (já 0-1)
    vocab = metrics.get('vocabulary_diversity', 0.5)
    normalized_metrics['vocabulary'] = max(0.0, min(1.0, vocab))
    
    # Coherence (0-100)
    coherence = metrics.get('coherence_score', 50) / 100.0
    normalized_metrics['coherence'] = max(0.0, min(1.0, coherence))
    
    # Originality (0-100)
    originality = metrics.get('originality_score', 50) / 100.0
    normalized_metrics['originality'] = max(0.0, min(1.0, originality))
    
    # Score ponderado
    weighted_score = sum(
        normalized_metrics[key] * weights.get(key, 0) 
        for key in normalized_metrics
    )
    
    # Converte para escala 0-100
    final_score = weighted_score * 100
    
    # Classificação de qualidade
    if final_score >= 90:
        category = "excellent"
        category_level = 5
    elif final_score >= 80:
        category = "very_good"
        category_level = 4
    elif final_score >= 70:
        category = "good"
        category_level = 3
    elif final_score >= 60:
        category = "fair"
        category_level = 2
    else:
        category = "poor"
        category_level = 1
    
    # Gera recomendações específicas
    recommendations = []
    threshold = 0.7
    
    if normalized_metrics['readability'] < threshold:
        recommendations.append({
            'area': 'readability',
            'issue': 'Text is difficult to read',
            'suggestion': 'Simplify sentence structure and use clearer vocabulary'
        })
    
    if normalized_metrics['grammar'] < 0.8:
        recommendations.append({
            'area': 'grammar',
            'issue': 'Grammar needs improvement',
            'suggestion': 'Review grammatical structures and punctuation'
        })
    
    if normalized_metrics['vocabulary'] < 0.6:
        recommendations.append({
            'area': 'vocabulary',
            'issue': 'Limited vocabulary diversity',
            'suggestion': 'Use more varied and precise terminology'
        })
    
    if normalized_metrics['coherence'] < threshold:
        recommendations.append({
            'area': 'coherence',
            'issue': 'Text lacks coherence',
            'suggestion': 'Improve logical flow and transitions between ideas'
        })
    
    if normalized_metrics['originality'] < 0.5:
        recommendations.append({
            'area': 'originality',
            'issue': 'Content lacks originality',
            'suggestion': 'Add more original insights and unique perspectives'
        })
    
    # Confidence baseada na variância das métricas
    values = list(normalized_metrics.values())
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    
    # Confiança maior para métricas consistentes
    consistency_factor = max(0.1, 1.0 - variance * 2)
    score_factor = min(1.0, final_score / 100.0)
    confidence = consistency_factor * 0.6 + score_factor * 0.4
    confidence = max(0.1, min(0.95, confidence))
    
    return {
        'score': float(final_score),
        'confidence': float(confidence),
        'quality_category': category,
        'category_level': category_level,
        'recommendations': recommendations,
        'metadata': {
            'normalized_metrics': normalized_metrics,
            'weights_used': weights,
            'variance': float(variance),
            'consistency_factor': float(consistency_factor),
            'total_recommendations': len(recommendations)
        }
    }


# ============================================================================
# BIOMATERIALS SCAFFOLD CONTRACT
# ============================================================================

def biomaterials_scaffold_contract(input_data: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Contrato Biomaterials Scaffold - Análise completa de scaffolds.
    
    Analisa estrutura de scaffold biomaterial usando métricas KEC,
    propriedades mecânicas e índices de biocompatibilidade.
    
    Args:
        input_data: Estrutura, propriedades materiais, rede de poros
        parameters: Configurações de análise
        
    Returns:
        Dict com métricas KEC, biocompatibilidade e propriedades
    """
    
    structure = input_data.get('scaffold_structure', {})
    materials = input_data.get('material_properties', {})
    pore_network = input_data.get('pore_network', {})
    
    if not structure or not materials:
        return {
            'error': 'Missing scaffold_structure or material_properties',
            'score': 0.0,
            'confidence': 0.0
        }
    
    # === MÉTRICAS ESTRUTURAIS ===
    
    porosity = float(structure.get('porosity', 0.5))
    connectivity = float(structure.get('connectivity', 0.7))
    pore_size_dist = structure.get('pore_size_distribution', [])
    surface_area = float(structure.get('surface_area_ratio', 1.0))
    
    # Valida porosity
    porosity = max(0.0, min(1.0, porosity))
    connectivity = max(0.0, min(1.0, connectivity))
    
    # === MÉTRICAS KEC ===
    
    # Entropia espectral baseada na distribuição de poros
    if pore_size_dist and sum(pore_size_dist) > 0:
        spectral_entropy = calculate_entropy(pore_size_dist)
    else:
        # Aproximação baseada na porosidade
        spectral_entropy = -porosity * safe_log(porosity, 2) - (1-porosity) * safe_log(1-porosity, 2)
    
    # Curvatura de Forman aproximada
    # Em grafos de poros: relacionada à conectividade e geometria
    forman_curvature_mean = connectivity * (1 - porosity) * surface_area
    
    # Small-world sigma (simplified)
    clustering_coeff = pore_network.get('clustering_coefficient', connectivity)
    path_length = pore_network.get('average_path_length', 1/connectivity if connectivity > 0 else 10)
    
    if path_length > 0:
        sigma = clustering_coeff / path_length
    else:
        sigma = 0.0
    
    # Small-world propensity
    swp = min(1.0, sigma / 2.0)  # Normaliza
    
    # === PROPRIEDADES MECÂNICAS ===
    
    young_modulus = float(materials.get('young_modulus', 1000))  # MPa
    tensile_strength = float(materials.get('tensile_strength', 50))  # MPa
    compressive_strength = float(materials.get('compressive_strength', tensile_strength * 3))
    
    # Normaliza propriedades mecânicas
    # Valores típicos para scaffolds: Young 100-5000 MPa, Tensile 10-200 MPa
    mechanical_score = min(1.0, (
        min(1.0, young_modulus / 2000.0) * 0.4 +
        min(1.0, tensile_strength / 100.0) * 0.3 +
        min(1.0, compressive_strength / 300.0) * 0.3
    ))
    
    # === BIOCOMPATIBILIDADE ===
    
    biocompatibility = float(materials.get('biocompatibility_index', 0.8))
    biocompatibility = max(0.0, min(1.0, biocompatibility))
    
    # Degradação controlada
    degradation_rate = float(materials.get('degradation_rate', 0.1))  # %/day
    degradation_score = max(0.0, 1.0 - degradation_rate * 10)  # Penaliza degradação rápida
    
    bio_score = (biocompatibility * 0.7 + degradation_score * 0.3)
    
    # === SCORE ESTRUTURAL ===
    
    # Combina porosidade, conectividade e entropia
    structural_score = (
        porosity * 0.3 +  # Porosidade adequada
        connectivity * 0.4 +  # Boa conectividade
        min(1.0, spectral_entropy / 5.0) * 0.3  # Entropia balanceada
    )
    
    # === SCORE TOPOLÓGICO ===
    
    topology_score = min(1.0, (forman_curvature_mean + swp) / 2)
    
    # === SCORE FINAL ===
    
    # Combina todas as dimensões
    final_score = (
        structural_score * 0.3 +
        mechanical_score * 0.25 +
        bio_score * 0.25 +
        topology_score * 0.2
    )
    
    # Índice de biocompatibilidade integrado
    biocompatibility_index = bio_score * structural_score * biocompatibility
    
    # Confidence baseada na completude dos dados
    data_completeness = (
        (1.0 if structure else 0.0) +
        (1.0 if materials else 0.0) +
        (0.5 if pore_network else 0.0) +
        (0.5 if pore_size_dist else 0.0)
    ) / 3.0
    
    confidence = min(0.95, final_score * 0.6 + data_completeness * 0.4)
    
    # Classificação do scaffold
    if final_score > 0.8:
        scaffold_rating = "excellent"
    elif final_score > 0.65:
        scaffold_rating = "good"
    elif final_score > 0.5:
        scaffold_rating = "acceptable"
    else:
        scaffold_rating = "inadequate"
    
    return {
        'score': float(final_score),
        'confidence': float(confidence),
        'scaffold_rating': scaffold_rating,
        'kec_metrics': {
            'H_spectral': float(spectral_entropy),
            'k_forman_mean': float(forman_curvature_mean),
            'sigma': float(sigma),
            'swp': float(swp)
        },
        'biocompatibility_index': float(biocompatibility_index),
        'mechanical_properties': {
            'structural_score': float(structural_score),
            'mechanical_score': float(mechanical_score),
            'young_modulus_normalized': min(1.0, young_modulus / 2000.0),
            'tensile_strength_normalized': min(1.0, tensile_strength / 100.0)
        },
        'topology_metrics': {
            'topology_score': float(topology_score),
            'clustering_coefficient': float(clustering_coeff),
            'average_path_length': float(path_length)
        },
        'metadata': {
            'porosity': porosity,
            'connectivity': connectivity,
            'surface_area_ratio': surface_area,
            'bio_score': float(bio_score),
            'data_completeness': float(data_completeness)
        }
    }


# ============================================================================
# NETWORK TOPOLOGY CONTRACT
# ============================================================================

def network_topology_contract(input_data: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Contrato Network Topology - Análise topológica de redes complexas.
    
    Analisa propriedades topológicas: densidade, clustering, centralidade,
    small-world, scale-free, modularidade.
    
    Args:
        input_data: Matriz de adjacência e atributos
        parameters: Configurações de análise
        
    Returns:
        Dict com métricas de rede e estrutura comunitária
    """
    
    adj_matrix = input_data.get('adjacency_matrix', [])
    node_attrs = input_data.get('node_attributes', {})
    analysis_type = input_data.get('analysis_type', 'full')
    
    if not adj_matrix:
        return {
            'error': 'Missing adjacency_matrix',
            'score': 0.0,
            'confidence': 0.0
        }
    
    n = len(adj_matrix)
    if n == 0:
        return {
            'error': 'Empty adjacency matrix',
            'score': 0.0,
            'confidence': 0.0
        }
    
    # Valida matriz quadrada
    if not all(len(row) == n for row in adj_matrix):
        return {
            'error': 'Adjacency matrix must be square',
            'score': 0.0,
            'confidence': 0.0
        }
    
    # === MÉTRICAS BÁSICAS ===
    
    # Conta arestas (assumindo grafo não-direcionado)
    total_edges = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j] > 0:
                total_edges += 1
    
    # Densidade da rede
    max_edges = n * (n - 1) / 2
    density = total_edges / max_edges if max_edges > 0 else 0
    
    # Distribuição de graus
    degrees = [sum(1 for j in range(n) if adj_matrix[i][j] > 0 and i != j) for i in range(n)]
    avg_degree = sum(degrees) / n if n > 0 else 0
    max_degree = max(degrees) if degrees else 0
    
    # === CLUSTERING COEFFICIENT ===
    
    clustering_coeffs = []
    total_clustering = 0
    
    for i in range(n):
        neighbors = [j for j in range(n) if adj_matrix[i][j] > 0 and i != j]
        k = len(neighbors)
        
        if k < 2:
            clustering_coeffs.append(0.0)
            continue
        
        # Conta triângulos
        triangles = 0
        for j in range(len(neighbors)):
            for l in range(j+1, len(neighbors)):
                if adj_matrix[neighbors[j]][neighbors[l]] > 0:
                    triangles += 1
        
        # Coeficiente de clustering local
        possible_triangles = k * (k - 1) / 2
        local_clustering = triangles / possible_triangles if possible_triangles > 0 else 0
        clustering_coeffs.append(local_clustering)
        total_clustering += local_clustering
    
    avg_clustering = total_clustering / n if n > 0 else 0
    
    # === PATH LENGTH APROXIMADO ===
    
    # Simplified BFS-based average path length
    total_path_length = 0
    path_count = 0
    
    for start in range(min(n, 10)):  # Sample first 10 nodes for efficiency
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            node, dist = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            if node != start:
                total_path_length += dist
                path_count += 1
            
            # Add neighbors
            for neighbor in range(n):
                if adj_matrix[node][neighbor] > 0 and neighbor not in visited:
                    queue.append((neighbor, dist + 1))
    
    avg_path_length = total_path_length / path_count if path_count > 0 else float('inf')
    
    # === CENTRALIDADE ===
    
    # Degree centrality
    degree_centrality = [deg / (n - 1) for deg in degrees] if n > 1 else [0] * n
    
    # Betweenness centrality aproximada (simplified)
    # Real implementation would need shortest paths calculation
    betweenness_centrality = [deg / max_degree for deg in degrees] if max_degree > 0 else [0] * n
    
    # === SMALL-WORLD ANALYSIS ===
    
    # Small-world sigma: C/C_random / L/L_random
    # C_random ≈ avg_degree/(n-1), L_random ≈ ln(n)/ln(avg_degree)
    c_random = avg_degree / (n - 1) if n > 1 else 0
    l_random = safe_log(n) / safe_log(max(avg_degree, 1)) if avg_degree > 1 else 1
    
    if c_random > 0 and l_random > 0 and avg_path_length != float('inf'):
        small_world_sigma = (avg_clustering / c_random) / (avg_path_length / l_random)
    else:
        small_world_sigma = 0.0
    
    # Small-world propensity (alternative measure)
    swp = avg_clustering / avg_path_length if avg_path_length > 0 and avg_path_length != float('inf') else 0
    
    # === MODULARIDADE APROXIMADA ===
    
    # Simplified modularity based on clustering
    modularity = avg_clustering * density  # Rough approximation
    
    # === SCORES ===
    
    # Connectivity score
    connectivity_score = min(1.0, density * 2)  # Penaliza redes muito esparsas
    
    # Efficiency score
    efficiency = avg_degree / max_degree if max_degree > 0 else 0
    efficiency_score = efficiency
    
    # Small-world score
    sw_score = min(1.0, small_world_sigma / 10.0) if small_world_sigma > 0 else 0
    
    # Modularity score
    modularity_score = min(1.0, modularity * 3)
    
    # Score final
    final_score = (
        connectivity_score * 0.3 +
        efficiency_score * 0.25 +
        sw_score * 0.25 +
        modularity_score * 0.2
    )
    
    # === DETECÇÃO DE COMUNIDADES ===
    
    # Estimativa baseada em clustering e densidade
    if avg_clustering > 0.3 and density < 0.3:
        estimated_communities = max(1, int(n * (1 - avg_clustering) / 3))
    else:
        estimated_communities = max(1, int(n / max(avg_degree, 1)))
    
    community_structure = {
        'estimated_communities': estimated_communities,
        'modularity': float(modularity),
        'avg_community_size': n / estimated_communities,
        'community_cohesion': avg_clustering
    }
    
    # Confidence baseada na conectividade e completude da análise
    confidence = min(0.95, (connectivity_score + efficiency_score) / 2)
    
    return {
        'score': float(final_score),
        'confidence': float(confidence),
        'network_metrics': {
            'density': float(density),
            'average_degree': float(avg_degree),
            'max_degree': max_degree,
            'clustering_coefficient': float(avg_clustering),
            'average_path_length': float(avg_path_length) if avg_path_length != float('inf') else None,
            'small_world_sigma': float(small_world_sigma),
            'small_world_propensity': float(swp),
            'efficiency': float(efficiency),
            'num_nodes': n,
            'num_edges': total_edges
        },
        'community_structure': community_structure,
        'centrality_measures': {
            'degree_centrality': [float(dc) for dc in degree_centrality[:10]],  # Top 10
            'betweenness_centrality': [float(bc) for bc in betweenness_centrality[:10]],
            'max_degree_centrality': float(max(degree_centrality)) if degree_centrality else 0,
            'centralization': (max_degree - avg_degree) / (n - 2) if n > 2 else 0
        },
        'metadata': {
            'analysis_type': analysis_type,
            'connectivity_score': float(connectivity_score),
            'efficiency_score': float(efficiency_score),
            'sw_score': float(sw_score),
            'modularity_score': float(modularity_score),
            'c_random': float(c_random),
            'l_random': float(l_random)
        }
    }


# ============================================================================
# SPECTRAL ANALYSIS CONTRACT
# ============================================================================

def spectral_analysis_contract(input_data: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Contrato Spectral Analysis - Análise espectral de grafos.
    
    Analisa propriedades espectrais do Laplaciano: eigenvalues,
    spectral gap, conectividade algébrica.
    
    Args:
        input_data: Laplaciano do grafo
        parameters: Configurações de análise espectral
        
    Returns:
        Dict com features espectrais e eigenvalues
    """
    
    laplacian = input_data.get('graph_laplacian', [])
    eigenvalue_analysis = input_data.get('eigenvalue_analysis', True)
    community_detection = input_data.get('community_detection', False)
    
    if not laplacian:
        return {
            'error': 'Missing graph_laplacian',
            'score': 0.0,
            'confidence': 0.0
        }
    
    n = len(laplacian)
    if n == 0:
        return {
            'error': 'Empty Laplacian matrix',
            'score': 0.0,
            'confidence': 0.0
        }
    
    # === ANÁLISE DA MATRIZ ===
    
    # Trace (soma dos eigenvalues)
    trace = sum(laplacian[i][i] for i in range(n))
    
    # Determinante aproximado (produto dos eigenvalues)
    # Para matrizes pequenas, calcula determinante simples
    if n <= 3:
        if n == 1:
            determinant = laplacian[0][0]
        elif n == 2:
            determinant = laplacian[0][0] * laplacian[1][1] - laplacian[0][1] * laplacian[1][0]
        else:  # n == 3
            determinant = (
                laplacian[0][0] * (laplacian[1][1] * laplacian[2][2] - laplacian[1][2] * laplacian[2][1]) -
                laplacian[0][1] * (laplacian[1][0] * laplacian[2][2] - laplacian[1][2] * laplacian[2][0]) +
                laplacian[0][2] * (laplacian[1][0] * laplacian[2][1] - laplacian[1][1] * laplacian[2][0])
            )
    else:
        determinant = 0.0  # Placeholder para matrizes grandes
    
    # === ESTIMATIVA DE EIGENVALUES ===
    
    # Usa teorema dos círculos de Gershgorin para estimar eigenvalues
    eigenvalue_estimates = []
    
    for i in range(n):
        center = laplacian[i][i]
        radius = sum(abs(laplacian[i][j]) for j in range(n) if i != j)
        eigenvalue_estimates.append({
            'min_estimate': center - radius,
            'max_estimate': center + radius,
            'center': center,
            'radius': radius
        })
    
    # Eigenvalues aproximados (centro dos círculos)
    approx_eigenvalues = [est['center'] for est in eigenvalue_estimates]
    approx_eigenvalues.sort()
    
    # === SPECTRAL GAP ===
    
    # O spectral gap é a diferença entre o segundo menor e o menor eigenvalue
    # Para Laplaciano de grafo conectado, menor eigenvalue = 0
    if len(approx_eigenvalues) >= 2:
        spectral_gap = approx_eigenvalues[1] - approx_eigenvalues[0]
        algebraic_connectivity = approx_eigenvalues[1]  # Second smallest
    else:
        spectral_gap = 0.0
        algebraic_connectivity = 0.0
    
    # Spectral radius (maior eigenvalue)
    spectral_radius = max(approx_eigenvalues) if approx_eigenvalues else 0.0
    
    # === FEATURES ESPECTRAIS ===
    
    spectral_features = {
        'spectral_gap': float(spectral_gap),
        'spectral_radius': float(spectral_radius),
        'algebraic_connectivity': float(algebraic_connectivity),
        'trace': float(trace),
        'estimated_determinant': float(determinant),
        'eigenvalue_spread': float(spectral_radius - approx_eigenvalues[0]) if approx_eigenvalues else 0.0
    }
    
    # === ANÁLISE DE CONECTIVIDADE ===
    
    # Conectividade baseada no spectral gap
    connectivity_score = min(1.0, algebraic_connectivity / (n / 4)) if n > 0 else 0
    
    # Balanceamento espectral
    if spectral_radius > 0:
        gap_ratio = spectral_gap / spectral_radius
        balance_score = min(1.0, gap_ratio * 4)  # Penaliza gap muito pequeno
    else:
        gap_ratio = 0.0
        balance_score = 0.0
    
    # === DETECÇÃO DE COMUNIDADES ===
    
    community_info = None
    if community_detection:
        # Número de comunidades baseado no spectral gap
        # Gap maior sugere fewer communities
        if spectral_gap > 0:
            estimated_communities = max(1, min(n//2, int(n / (spectral_gap + 1))))
        else:
            estimated_communities = n  # Cada nó é uma comunidade
        
        community_info = {
            'estimated_communities': estimated_communities,
            'community_strength': float(gap_ratio),
            'separation_quality': min(1.0, spectral_gap)
        }
    
    # === SCORE FINAL ===
    
    # Combina conectividade e balanceamento espectral
    final_score = connectivity_score * 0.6 + balance_score * 0.4
    
    # Confidence baseada na magnitude do spectral gap
    if spectral_radius > 0:
        confidence = min(0.95, spectral_gap / (spectral_radius + 0.1))
    else:
        confidence = 0.1
    
    # Classificação da rede
    if final_score > 0.8:
        network_type = "well_connected"
    elif final_score > 0.6:
        network_type = "moderately_connected"
    elif final_score > 0.3:
        network_type = "poorly_connected"
    else:
        network_type = "disconnected"
    
    return {
        'score': float(final_score),
        'confidence': float(confidence),
        'network_type': network_type,
        'spectral_features': spectral_features,
        'eigenvalues': [float(e) for e in approx_eigenvalues[:min(10, n)]],
        'community_structure': community_info,
        'connectivity_analysis': {
            'connectivity_score': float(connectivity_score),
            'balance_score': float(balance_score),
            'gap_ratio': float(gap_ratio)
        },
        'metadata': {
            'matrix_size': n,
            'analysis_type': analysis_type,
            'eigenvalue_method': 'gershgorin_estimation',
            'avg_degree': float(avg_degree),
            'density': float(density)
        }
    }


# ============================================================================
# CONTRACT DISPATCHER
# ============================================================================

def execute_mathematical_contract(
    contract_type: str,
    input_data: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Dispatcher principal para execução de contratos matemáticos.
    
    Args:
        contract_type: Tipo do contrato
        input_data: Dados de entrada
        parameters: Parâmetros opcionais
        
    Returns:
        Resultado da execução do contrato
    """
    
    contract_functions = {
        'delta_kec_v1': delta_kec_v1_contract,
        'zuco_reading_v1': zuco_reading_v1_contract,
        'editorial_v1': editorial_v1_contract,
        'biomaterials_scaffold': biomaterials_scaffold_contract,
        'network_topology': network_topology_contract,
        'spectral_analysis': spectral_analysis_contract
    }
    
    contract_func = contract_functions.get(contract_type)
    if not contract_func:
        return {
            'error': f'Unknown contract type: {contract_type}',
            'available_types': list(contract_functions.keys()),
            'score': 0.0,
            'confidence': 0.0
        }
    
    try:
        result = contract_func(input_data, parameters)
        
        # Garante campos obrigatórios
        if 'score' not in result:
            result['score'] = 0.0
        if 'confidence' not in result:
            result['confidence'] = 0.0
        
        # Adiciona metadados de execução
        result.setdefault('metadata', {}).update({
            'contract_type': contract_type,
            'execution_method': 'mathematical_contract',
            'parameters_used': parameters or {}
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Mathematical contract execution failed: {e}")
        return {
            'error': f'Contract execution failed: {str(e)}',
            'contract_type': contract_type,
            'score': 0.0,
            'confidence': 0.0,
            'metadata': {
                'error_type': type(e).__name__,
                'parameters_used': parameters or {}
            }
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_mathematical_contracts() -> List[Dict[str, Any]]:
    """Retorna lista de contratos matemáticos disponíveis."""
    
    contracts = [
        {
            'type': 'delta_kec_v1',
            'name': 'Delta Knowledge Exchange Coefficient v1',
            'description': 'Calcula transferência de conhecimento entre sistemas',
            'complexity': 'medium',
            'computational_cost': 'low'
        },
        {
            'type': 'zuco_reading_v1',
            'name': 'ZuCo Reading Comprehension v1',
            'description': 'Análise de compreensão baseada em EEG e eye tracking',
            'complexity': 'high',
            'computational_cost': 'medium'
        },
        {
            'type': 'editorial_v1',
            'name': 'Editorial Quality Assessment v1',
            'description': 'Avaliação abrangente de qualidade textual',
            'complexity': 'medium',
            'computational_cost': 'low'
        },
        {
            'type': 'biomaterials_scaffold',
            'name': 'Biomaterials Scaffold Analysis',
            'description': 'Análise matemática completa de scaffolds biomateriais',
            'complexity': 'very_high',
            'computational_cost': 'high'
        },
        {
            'type': 'network_topology',
            'name': 'Network Topology Analysis',
            'description': 'Análise topológica de redes complexas',
            'complexity': 'high',
            'computational_cost': 'high'
        },
        {
            'type': 'spectral_analysis',
            'name': 'Spectral Graph Analysis',
            'description': 'Análise espectral de grafos e redes',
            'complexity': 'very_high',
            'computational_cost': 'very_high'
        }
    ]
    
    return contracts


def validate_contract_input(contract_type: str, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valida entrada para contrato matemático.
    
    Args:
        contract_type: Tipo do contrato
        input_data: Dados de entrada
        
    Returns:
        Tuple (is_valid, errors)
    """
    
    errors = []
    
    if contract_type == 'delta_kec_v1':
        if 'source_entropy' not in input_data:
            errors.append("Missing required field: source_entropy")
        if 'target_entropy' not in input_data:
            errors.append("Missing required field: target_entropy")
    
    elif contract_type == 'zuco_reading_v1':
        if 'eeg_features' not in input_data:
            errors.append("Missing required field: eeg_features")
        if 'eye_tracking_features' not in input_data:
            errors.append("Missing required field: eye_tracking_features")
    
    elif contract_type == 'editorial_v1':
        if 'text_metrics' not in input_data:
            errors.append("Missing required field: text_metrics")
    
    elif contract_type == 'biomaterials_scaffold':
        if 'scaffold_structure' not in input_data:
            errors.append("Missing required field: scaffold_structure")
        if 'material_properties' not in input_data:
            errors.append("Missing required field: material_properties")
    
    elif contract_type == 'network_topology':
        if 'adjacency_matrix' not in input_data:
            errors.append("Missing required field: adjacency_matrix")
    
    elif contract_type == 'spectral_analysis':
        if 'graph_laplacian' not in input_data:
            errors.append("Missing required field: graph_laplacian")
    
    else:
        errors.append(f"Unknown contract type: {contract_type}")
    
    return len(errors) == 0, errors