"""
MolGuard — Yapılandırma Yöneticisi / Configuration Manager
config.yaml dosyasını okur / Reads config.yaml
"""

import yaml
import os


def load_config(config_path='src/utils/config.yaml'):
    """Yapılandırma dosyasını yükle / Load configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Yapılandırma bulunamadı / Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_config(config, model_name):
    """Model yapılandırmasını al / Get model configuration"""
    return config.get('models', {}).get(model_name, {})


def get_training_config(config):
    """Eğitim yapılandırmasını al / Get training configuration"""
    return config.get('training', {})
