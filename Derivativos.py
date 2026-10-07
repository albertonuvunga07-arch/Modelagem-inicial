import numpy as np
from scipy.stats import norm

def black_scholes_call(S_0, K, T, r, sigma):
    """
    Calcula o preço exato (analítico) de uma Opção Call Europeia usando Black-Scholes.
    
    Parâmetros:
    S_0   : Preço atual do ativo subjacente
    K     : Preço de exercício (Strike)
    T     : Tempo até o vencimento (em anos)
    r     : Taxa de juros livre de risco (anualizada)
    sigma : Volatilidade do ativo (anualizada)
    """
    # Cálculo das variáveis d1 e d2
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Preço da Call = S_0 * N(d1) - K * e^(-rT) * N(d2)
    call_price = S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price, d1, d2


def monte_carlo_call(S_0, K, T, r, sigma, num_paths=500000, seed=42):
    """
    Estima o preço de uma Opção Call Europeia via Simulação de Monte Carlo.
    """
    np.random.seed(seed)
    
    # 1. Gerar choques aleatórios Z da Distribuição Normal Padrão N(0, 1)
    Z = np.random.normal(0.0, 1.0, num_paths)
    
    # 2. Simular o preço do ativo no vencimento T (S_T) sob Movimento Browniano Geométrico
    S_T = S_0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # 3. Calcular o Payoff da Call no vencimento: max(S_T - K, 0)
    payoffs = np.maximum(S_T - K, 0)
    
    # 4. Descontar os Payoffs ao valor presente usando e^(-rT)
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    # 5. Preço da opção = Média dos payoffs descontados
    mc_price = np.mean(discounted_payoffs)
    
    # Erro Padrão da Simulação (Mede a precisão da amostragem)
    std_error = np.std(discounted_payoffs) / np.sqrt(num_paths)
    
    return mc_price, std_error


# ==============================================================================
# CONFIGURAÇÃO DOS PARÂMETROS FINANCEIROS
# ==============================================================================
S_0   = 100.0   # Preço inicial da ação ($100.00)
K     = 100.0   # Preço de exercício (At-the-Money)
T     = 1.0     # Vencimento em 1 ano
r     = 0.05    # Taxa livre de risco (5% ao ano)
sigma = 0.20    # Volatilidade (20% ao ano)

# Execução dos dois métodos
bs_price, d1, d2 = black_scholes_call(S_0, K, T, r, sigma)
mc_price, std_err = monte_carlo_call(S_0, K, T, r, sigma, num_paths=500000)

# Exibição dos Resultados
print("=" * 55)
print("            PREÇAMENTO DE OPÇÃO CALL EUROPEIA           ")
print("=" * 55)
print(f"Parâmetros: S0=${S_0:.2f} | K=${K:.2f} | T={T} ano | r={r*100:.1f}% | σ={sigma*100:.1f}%\n")

print("1. MÉTODO DE BLACK-SCHOLES (Solução Analítica Exata)")
print(f"   d1 = {d1:.4f}  |  d2 = {d2:.4f}")
print(f"   Preço Justo da Call (BS) = ${bs_price:.4f}\n")

print("2. SIMULAÇÃO DE MONTE CARLO (500.000 Caminhos)")
print(f"   Preço Estimado da Call (MC) = ${mc_price:.4f}")
print(f"   Erro Padrão da Estimativa   = ±${std_err:.4f}")
print(f"   Intervalo de Confiança 95%  = [${mc_price - 1.96*std_err:.4f}, ${mc_price + 1.96*std_err:.4f}]")
print(f"   Diferença Relativa vs BS    = {abs(mc_price - bs_price) / bs_price * 100:.3f}%")
print("=" * 55)