import numpy as np
from scipy.stats import skewnorm, t

def simulate_crypto_price_paths(
    current_price, time_increment, time_length, num_simulations, asset, ai_params=None
):
    """
    Simulate multiple crypto asset price paths.
    If ai_params is provided, use AI-predicted interval parameters (arrays) instead of defaults.
    
    ai_params format:
    - For interval prediction: {'sigma': array[288], 'skewness': array[288], 'kurtosis': array[288]}
    - For legacy single values: {'daily_sigma': float, 'skewness': float, 'kurtosis': float}
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

    # Check if we have interval-based AI parameters
    if ai_params and 'sigma' in ai_params and isinstance(ai_params['sigma'], np.ndarray):
        # New interval-based parameters (288 values each)
        price_paths = []
        for _ in range(num_simulations):
            price_path = simulate_single_price_path_intervals(
                current_price, time_increment, time_length, ai_params
            )
            price_paths.append(price_path)
        return np.array(price_paths)
    
    else:
        # Legacy single-value parameters or defaults
        if ai_params and asset == 'BTC':
            params = CURRENT_DAILY_PARAMS[asset].copy()
            params.update(ai_params)
        else:
            params = CURRENT_DAILY_PARAMS[asset]

        price_paths = []
        for _ in range(num_simulations):
            price_path = simulate_single_price_path_daily(
                current_price, time_increment, time_length, **params
            )
            price_paths.append(price_path)

        return np.array(price_paths)


def simulate_single_price_path_daily(
    current_price, 
    time_increment, 
    time_length, 
    daily_sigma,                    # DAILY volatility (not annual)
    daily_drift=0.0,               # Expected DAILY return
    skewness=0.0,                  # Distribution skewness
    kurtosis=3.0,                  # Distribution kurtosis
    volatility_clustering=False,   # GARCH-like volatility
    min_price=0.01                 # Price floor
):
    """
    Simulate price path using DAILY parameters (no mean reversion, no jumps).
    
    Parameters:
    -----------
    daily_sigma : float
        Daily volatility (e.g., 0.037 for 3.7% daily volatility)
    daily_drift : float  
        Expected daily return (e.g., 0.001 for 0.1% daily)
    """
    
    # Time calculations
    one_day = 86400  # seconds in a day
    dt = time_increment / one_day  # Time step as fraction of day
    num_steps = int(time_length / time_increment)
    
    # Initialize arrays
    prices = np.zeros(num_steps + 1)
    prices[0] = current_price
    volatilities = np.full(num_steps, daily_sigma)
    
    # Generate random numbers
    if abs(skewness) > 0.1 or abs(kurtosis - 3) > 0.5:
        random_nums = generate_skewed_kurtotic_random(num_steps, skewness, kurtosis)
    else:
        random_nums = np.random.normal(0, 1, size=num_steps)
    
    # Main simulation loop
    for i in range(num_steps):
        current_vol = volatilities[i]
        # 1. Base return with daily drift and volatility
        drift_component = daily_drift * dt
        vol_component = current_vol * np.sqrt(dt) * random_nums[i]
        total_return = drift_component + vol_component
        # 2. Update price
        new_price = prices[i] * (1 + total_return)
        prices[i + 1] = max(new_price, min_price)
        # 3. Update volatility (daily clustering)
        if volatility_clustering and i < num_steps - 1:
            volatilities[i + 1] = update_volatility_garch_daily(
                current_vol, total_return, daily_sigma
            )
    
    return prices

def simulate_single_price_path_intervals(
    current_price, 
    time_increment, 
    time_length, 
    ai_params
):
    """
    Simulate price path using interval-specific parameters.
    
    Parameters:
    -----------
    ai_params : dict
        Contains 'sigma', 'skewness', 'kurtosis' arrays with 288 elements each
    """
    
    # Time calculations
    num_steps = int(time_length / time_increment)
    
    # Get parameter arrays
    sigma_array = ai_params['sigma']
    skewness_array = ai_params['skewness'] 
    kurtosis_array = ai_params['kurtosis']
    
    # Ensure we have enough parameters for the simulation
    if len(sigma_array) < num_steps:
        # Repeat the pattern if needed
        repeat_factor = (num_steps // len(sigma_array)) + 1
        sigma_array = np.tile(sigma_array, repeat_factor)[:num_steps]
        skewness_array = np.tile(skewness_array, repeat_factor)[:num_steps]
        kurtosis_array = np.tile(kurtosis_array, repeat_factor)[:num_steps]
    else:
        # Truncate if we have more parameters than needed
        sigma_array = sigma_array[:num_steps]
        skewness_array = skewness_array[:num_steps]
        kurtosis_array = kurtosis_array[:num_steps]
    
    # Initialize arrays
    prices = np.zeros(num_steps + 1)
    prices[0] = current_price
    
    # Main simulation loop with interval-specific parameters
    for i in range(num_steps):
        current_sigma = sigma_array[i]
        current_skewness = skewness_array[i] 
        current_kurtosis = kurtosis_array[i]
        
        # Generate random number with current distribution characteristics
        if abs(current_skewness) > 0.1 or abs(current_kurtosis - 3) > 0.5:
            random_num = generate_skewed_kurtotic_single(current_skewness, current_kurtosis)
        else:
            random_num = np.random.normal(0, 1)
        
        # Calculate price change using current interval's volatility
        vol_component = current_sigma * random_num
        
        # Update price (no drift assumed for 5-minute intervals)
        new_price = prices[i] * (1 + vol_component)
        prices[i + 1] = max(new_price, 1000)  # Apply minimum price floor
    
    return prices

def generate_skewed_kurtotic_single(skewness, kurtosis):
    """Generate a single random number with specified skewness and kurtosis"""
    if abs(skewness) > 0.1:
        # Use skewed normal distribution
        a = skewness  # shape parameter
        return skewnorm.rvs(a)
    elif abs(kurtosis - 3) > 0.5:
        # Use t-distribution for excess kurtosis
        # Convert kurtosis to degrees of freedom
        if kurtosis > 3:
            df = max(2.1, 6 / (kurtosis - 3))  # Ensure df > 2
            return t.rvs(df)
        else:
            return np.random.normal(0, 1)
    else:
        return np.random.normal(0, 1)

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
