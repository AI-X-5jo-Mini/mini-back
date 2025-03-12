import numpy as np

def calculate_distance(p1, p2):
    """두 랜드마크 좌표 간의 거리 계산"""
    return np.linalg.norm(np.array(p1) - np.array(p2))
