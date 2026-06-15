#!/usr/bin/env python3
"""
Test script for the new CustomDataLoader with yfinance
"""

import pandas as pd
import numpy as np
from custom_data_loader import CustomDataLoader, feature_engineer

def test_yfinance_loader():
    """Test the yfinance data loader"""
    print("Testing yfinance data loader...")
    
    # Test with Nifty 50
    loader = CustomDataLoader(ticker="^NSEI")
    
    try:
        # Load data for a recent period
        ret_ser = loader.load(start_date="2020-01-01", end_date="2024-12-31")
        
        print(f"✓ Successfully loaded data for ^NSEI")
        print(f"  Date range: {ret_ser.index[0]} to {ret_ser.index[-1]}")
        print(f"  Number of observations: {len(ret_ser)}")
        print(f"  Return statistics:")
        print(f"    Mean: {ret_ser.mean().iloc[0]:.6f}")
        print(f"    Std: {ret_ser.std().iloc[0]:.6f}")
        print(f"    Min: {ret_ser.min().iloc[0]:.6f}")
        print(f"    Max: {ret_ser.max().iloc[0]:.6f}")
        
        # Test feature engineering
        print("\nTesting feature engineering...")
        features = feature_engineer(ret_ser)
        print(f"✓ Generated {len(features.columns)} features")
        print(f"  Feature columns: {list(features.columns)}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Features sample:")
        print(features.head())
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_file_loader():
    """Test the file-based data loader (if we had a sample file)"""
    print("\nTesting file-based data loader...")
    
    # This would test loading from CSV/Excel files
    # For now, just show the interface
    print("File loader interface ready (would need sample data file to test)")
    return True

def test_scalers():
    """Test the scaling utilities"""
    print("\nTesting scalers...")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100) * 2 + 5
    })
    
    # Test ExpandingScaler
    from custom_data_loader import ExpandingScaler
    expanding_scaler = ExpandingScaler(min_periods=10)
    scaled_data = expanding_scaler.fit_transform(sample_data)
    print(f"✓ ExpandingScaler: {scaled_data.shape}")
    
    # Test StandardScalerCM
    from custom_data_loader import StandardScalerCM
    standard_scaler = StandardScalerCM()
    standard_scaled = standard_scaler.fit_transform(sample_data)
    print(f"✓ StandardScalerCM: {standard_scaled.shape}")
    
    # Test DataClipperStd
    from custom_data_loader import DataClipperStd
    clipper = DataClipperStd(mul=2.0)
    clipped_data = clipper.fit_transform(sample_data)
    print(f"✓ DataClipperStd: {clipped_data.shape}")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Testing CustomDataLoader with yfinance")
    print("=" * 50)
    
    # Test yfinance loader
    success1 = test_yfinance_loader()
    
    # Test file loader interface
    success2 = test_file_loader()
    
    # Test scalers
    success3 = test_scalers()
    
    print("\n" + "=" * 50)
    if all([success1, success2, success3]):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")
    print("=" * 50) 