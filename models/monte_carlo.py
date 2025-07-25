import numpy as np
from scipy.stats import skewnorm, t

def simulate_crypto_price_paths(
    current_price, time_increment, time_length, num_simulations, asset, ai_params=None
):
    """
    Simulate multiple crypto asset price paths.
    If ai_params is provided, use AI-predicted parameters instead of defaults.
    """

    CURRENT_DAILY_PARAMS = {
        'BTC': {
            # Core volatility and drift
            'daily_sigma': 0.0366,              # 3.66% daily volatility
            'daily_drift': 0.00,              # 0.04% daily expected return (~15% annual)
            # Distribution characteristics
            'skewness': 0.09,                  # Negative skew (crash risk)
            'kurtosis': 1.34,                    # Fat tails (extreme events)
            # Advanced features
            'volatility_clustering': True,
            'min_price': 1000                   # $1k floor price
        },
        
        'ETH': {
            # Core volatility and drift  
            'daily_sigma': 0.0419,              # 4.19% daily volatility
            'daily_drift': 0.0,              # 0.05% daily expected return (~20% annual)
            # Distribution characteristics
            'skewness': -0.2,                  # More negative skew than BTC
            'kurtosis': 4.5,                    # Fatter tails than BTC
            # Advanced features
            'volatility_clustering': True,
            'min_price': 100                    # $100 floor price
        },
        
        'XAU': {
            # Core volatility and drift
            'daily_sigma': 0.0105,              # 1.05% daily volatility
            'daily_drift': 0.00,              # 0.01% daily expected return (~4% annual)
            # Distribution characteristics  
            'skewness': 0.0,                   # Slight positive skew (safe haven flows)
            'kurtosis': 0.5,                    # Slightly fat tails
            # Advanced features
            'volatility_clustering': False,      # Less clustering than crypto
            'min_price': 1000                   # $1000/oz floor
        }
    }

    # Use AI-predicted parameters if available, otherwise use defaults
    if ai_params and asset == 'BTC':
        params = CURRENT_DAILY_PARAMS[asset].copy()
        params.update(ai_params)
    else:
        params = CURRENT_DAILY_PARAMS[asset]

    price_paths = []
    for _ in range(num_simulations):
        price_path = simulate_single_price_path_multistep(
            current_price, time_increment, time_length,
            sigma=params.get('sigma', params['daily_sigma']),
            skewness=params.get('skewness', params['skewness']),
            kurtosis=params.get('kurtosis', params['kurtosis']),
            min_price=params.get('min_price', 0.01)
        )
        price_paths.append(price_path)

    return np.array(price_paths)


def simulate_single_price_path_multistep(
    current_price,
    time_increment,
    time_length,
    sigma,
    skewness,
    kurtosis,
    min_price=0.01
):
    """
    Simulate price path using per-step parameters (sigma, skewness, kurtosis can be arrays or scalars).
    """
    one_day = 86400
    num_steps = int(time_length / time_increment)
    prices = np.zeros(num_steps + 1)
    prices[0] = current_price
    # Broadcast if needed
    if np.isscalar(sigma):
        sigma = np.full(num_steps, sigma)
    if np.isscalar(skewness):
        skewness = np.full(num_steps, skewness)
    if np.isscalar(kurtosis):
        kurtosis = np.full(num_steps, kurtosis)
    dt = time_increment / one_day
    for i in range(num_steps):
        # Generate random number for this step
        if abs(skewness[i]) > 0.1 or abs(kurtosis[i] - 3) > 0.5:
            rand = generate_skewed_kurtotic_random(1, skewness[i], kurtosis[i])[0]
        else:
            rand = np.random.normal(0, 1)
        drift_component = 0.0  # No drift for each step (or add if needed)
        vol_component = sigma[i] * np.sqrt(dt) * rand
        total_return = drift_component + vol_component
        new_price = prices[i] * (1 + total_return)
        prices[i + 1] = max(new_price, min_price)
    return prices

def generate_skewed_kurtotic_random(size, skewness, kurtosis):
    """Generate random numbers with skewness and kurtosis"""
    if abs(skewness) > 0.1:
        return skewnorm.rvs(a=skewness, size=size)
    elif kurtosis > 4:
        df = 4 + 6 / max(kurtosis - 3, 0.1)
        return t.rvs(df=df, size=size)
    else:
        return np.random.normal(0, 1, size=size)

def update_volatility_garch_daily(current_vol, recent_return, base_vol, alpha=0.1, beta=0.85):
    """Update volatility using daily GARCH"""
    omega = base_vol * base_vol * (1 - alpha - beta)
    new_variance = omega + alpha * (recent_return * recent_return) + beta * (current_vol * current_vol)
    return np.sqrt(max(new_variance, 0.001))
