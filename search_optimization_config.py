"""
Recommended configuration updates for better search precision
"""

# Recommended threshold configuration for production
RECOMMENDED_SEARCH_THRESHOLDS = {
    # Conservative thresholds (high precision, some recall loss)
    "conservative": {
        "title_similarity_threshold": 0.4,
        "description_similarity_threshold": 0.4, 
        "document_similarity_threshold": 0.4
    },
    
    # Balanced thresholds (good precision/recall balance)
    "balanced": {
        "title_similarity_threshold": 0.35,
        "description_similarity_threshold": 0.35,
        "document_similarity_threshold": 0.35
    },
    
    # Liberal thresholds (high recall, some precision loss)
    "liberal": {
        "title_similarity_threshold": 0.25,
        "description_similarity_threshold": 0.25,
        "document_similarity_threshold": 0.25
    }
}

# Enhanced search parameters for better precision
ENHANCED_SEARCH_PARAMS = {
    # Use exact phrase boosting
    "exact_phrase_boost": 0.3,
    
    # Increase search scope for better aggregation
    "search_multiplier": 3,  # Search for limit * 3 candidates before filtering
    
    # Better index parameters for precision
    "index_params": {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT", 
        "params": {"nlist": 256}  # Increased for better precision
    },
    
    # Search parameters for better recall
    "search_params": {
        "metric_type": "COSINE",
        "params": {"nprobe": 20}  # Increased from 10 for better recall
    }
}
