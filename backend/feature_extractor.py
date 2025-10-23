"""
Feature extraction module for PhishNet.
Extracts features from URLs and email content for phishing detection.
"""
import re
from urllib.parse import urlparse
from typing import Dict, List


class FeatureExtractor:
    """Extract features from URLs and emails for ML classification."""
    
    @staticmethod
    def extract_url_features(url: str) -> Dict[str, float]:
        """
        Extract features from a URL for phishing detection.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # URL length
        features['url_length'] = len(url)
        
        # Number of dots in URL
        features['num_dots'] = url.count('.')
        
        # Number of hyphens
        features['num_hyphens'] = url.count('-')
        
        # Number of underscores
        features['num_underscores'] = url.count('_')
        
        # Number of slashes
        features['num_slashes'] = url.count('/')
        
        # Number of question marks
        features['num_questions'] = url.count('?')
        
        # Number of equal signs
        features['num_equals'] = url.count('=')
        
        # Number of at symbols
        features['num_at'] = url.count('@')
        
        # Number of ampersands
        features['num_ampersands'] = url.count('&')
        
        # Number of digits
        features['num_digits'] = sum(c.isdigit() for c in url)
        
        # Parse URL
        try:
            parsed = urlparse(url)
            
            # Has IP address instead of domain name
            ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            features['has_ip'] = 1.0 if re.search(ip_pattern, parsed.netloc) else 0.0
            
            # HTTPS vs HTTP
            features['is_https'] = 1.0 if parsed.scheme == 'https' else 0.0
            
            # Domain length
            features['domain_length'] = len(parsed.netloc)
            
            # Path length
            features['path_length'] = len(parsed.path)
            
            # Query length
            features['query_length'] = len(parsed.query)
            
            # Has suspicious keywords
            suspicious_keywords = ['login', 'signin', 'account', 'update', 'verify', 
                                 'secure', 'bank', 'confirm', 'password']
            features['has_suspicious_keywords'] = 1.0 if any(
                keyword in url.lower() for keyword in suspicious_keywords
            ) else 0.0
            
        except Exception:
            # If parsing fails, set default values
            features['has_ip'] = 0.0
            features['is_https'] = 0.0
            features['domain_length'] = 0
            features['path_length'] = 0
            features['query_length'] = 0
            features['has_suspicious_keywords'] = 0.0
        
        return features
    
    @staticmethod
    def extract_email_features(email_content: str, sender: str = "") -> Dict[str, float]:
        """
        Extract features from email content for phishing detection.
        
        Args:
            email_content: The email body text
            sender: The sender email address
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Email content length
        features['content_length'] = len(email_content)
        
        # Number of URLs in email
        # Use a more specific pattern to count URLs (simple pattern for phishing detection)
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, email_content)
        features['num_urls'] = len(urls)
        
        # Number of suspicious keywords
        suspicious_keywords = ['urgent', 'verify', 'account', 'suspended', 'click', 
                             'confirm', 'security', 'update', 'password', 'credit card']
        features['num_suspicious_keywords'] = sum(
            email_content.lower().count(keyword) for keyword in suspicious_keywords
        )
        
        # Has money-related terms
        money_keywords = ['$', 'money', 'prize', 'lottery', 'winner', 'payment']
        features['has_money_keywords'] = 1.0 if any(
            keyword in email_content.lower() for keyword in money_keywords
        ) else 0.0
        
        # Number of exclamation marks
        features['num_exclamations'] = email_content.count('!')
        
        # Number of capital letters (normalized)
        num_capitals = sum(1 for c in email_content if c.isupper())
        features['capital_ratio'] = num_capitals / len(email_content) if len(email_content) > 0 else 0.0
        
        # Has attachments mentioned
        attachment_keywords = ['attachment', 'attached', 'download', 'file']
        features['mentions_attachments'] = 1.0 if any(
            keyword in email_content.lower() for keyword in attachment_keywords
        ) else 0.0
        
        # Sender domain features
        if sender:
            features['sender_length'] = len(sender)
            features['sender_has_numbers'] = 1.0 if any(c.isdigit() for c in sender) else 0.0
        else:
            features['sender_length'] = 0
            features['sender_has_numbers'] = 0.0
        
        return features
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Return list of all possible feature names."""
        url_features = [
            'url_length', 'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
            'num_questions', 'num_equals', 'num_at', 'num_ampersands', 'num_digits',
            'has_ip', 'is_https', 'domain_length', 'path_length', 'query_length',
            'has_suspicious_keywords'
        ]
        email_features = [
            'content_length', 'num_urls', 'num_suspicious_keywords', 'has_money_keywords',
            'num_exclamations', 'capital_ratio', 'mentions_attachments', 'sender_length',
            'sender_has_numbers'
        ]
        return url_features + email_features
