
# coding: utf-8

# In[13]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# 参数设置
N = 5  # 小基站数量
M = 10  # 短视频数量
p = 3   # 每个小基站最大缓存容量

def generate_requests_matrix(N, M, low=1, high=10):
    return np.random.randint(low, high, (N, M))  # 生成用户请求矩阵

def create_network_topology(N):
    return nx.complete_graph(N)  # 创建完全图拓扑

def initialize_cache_strategy(N, M, p):
    cache_strategy = np.zeros((N, M), dtype=int)  # 初始化缓存策略矩阵
    for i in range(N):
        cache_strategy[i, np.random.choice(M, p, replace=False)] = 1  # 随机选择一些短视频进行缓存
    return cache_strategy

def calculate_utility(i, a, requests, G, alpha, beta, gamma):
    local_served_requests = 0
    one_hop_served_requests = 0
    cloud_served_requests = 0
    
    for r in range(M):
        if a[i][r] == 1:  # 如果小基站i缓存了视频r
            local_served_requests += requests[i][r]  # 增加本地服务请求计数
        else:  # 如果小基站i没有缓存视频r
            cached_by_others = any(a[j][r] == 1 for j in range(N) if j != i)  # 检查其他小基站是否缓存了视频r
            if not cached_by_others:
                cloud_served_requests += requests[i][r]  # 增加云端服务请求计数
            else:
                nearest_hops = [len(nx.shortest_path(G, source=i, target=j)) - 1 for j in range(N) if j != i and a[j][r] == 1]  # 计算最近跳数
                if nearest_hops:
                    nearest_hop = min(nearest_hops)  # 找到最小跳数
                    one_hop_served_requests += requests[i][r]  # 增加一跳服务请求计数
                else:
                    cloud_served_requests += requests[i][r]  # 增加云端服务请求计数
    
    utility = (local_served_requests * alpha) - (one_hop_served_requests * beta) - (cloud_served_requests * gamma)  # 计算效用
    return utility

def update_cache_strategy(cache_strategy, requests, G, alpha, beta, gamma, max_iterations=100):
    converged = False
    iteration = 0
    
    while not converged and iteration < max_iterations:
        converged = True
        new_cache_strategy = cache_strategy.copy()
        
        for i in range(N):
            best_utility = float('-inf')
            best_strategy = None
            
            for combination in combinations(range(M), p):  # 遍历所有可能的缓存组合
                temp_strategy = np.zeros(M, dtype=int)
                temp_strategy[list(combination)] = 1
                
                updated_cache_strategy = np.vstack([temp_strategy if idx == i else row for idx, row in enumerate(cache_strategy)])  # 更新缓存策略
                
                current_utility = calculate_utility(i, cache_strategy, requests, G, alpha, beta, gamma)  # 计算当前效用
                temp_strategy_utility = calculate_utility(i, updated_cache_strategy, requests, G, alpha, beta, gamma)  # 计算新策略效用
                
                m_r = temp_strategy_utility - current_utility  # 计算效用差异
                
                if m_r > best_utility:
                    best_utility = m_r
                    best_strategy = temp_strategy
                    
            if not np.array_equal(best_strategy, cache_strategy[i]):  # 如果找到更好的策略
                converged = False
                new_cache_strategy[i] = best_strategy
        
        cache_strategy = new_cache_strategy
        iteration += 1
    
    print("Converged after", iteration, "iterations")  # 输出收敛迭代次数
    return cache_strategy

def evaluate_performance(final_cache_strategy, requests, G, alpha, beta, gamma):
    initial_latency = sum(sum(requests)) * gamma  # 初始延迟
    initial_traffic = sum(sum(requests)) * gamma  # 初始流量
    initial_cloud_workload = sum(sum(requests))  # 初始云工作负载
    
    total_requests = sum(sum(requests))
    local_served_requests = 0
    one_hop_served_requests = 0
    cloud_served_requests = 0
    total_latency = 0
    final_traffic = 0
    
    for i in range(N):
        for r in range(M):
            if final_cache_strategy[i][r] == 1:  # 如果小基站i缓存了视频r
                local_served_requests += requests[i][r]  # 增加本地服务请求计数
                total_latency += requests[i][r] * alpha  # 增加总延迟
                final_traffic += requests[i][r] * alpha  # 增加最终流量
            else:  # 如果小基站i没有缓存视频r
                cached_by_others = any(final_cache_strategy[j][r] == 1 for j in range(N) if j != i)  # 检查其他小基站是否缓存了视频r
                if cached_by_others:
                    nearest_hop = min([len(nx.shortest_path(G, source=i, target=j)) - 1 for j in range(N) if j != i and final_cache_strategy[j][r] == 1])  # 计算最近跳数
                    one_hop_served_requests += requests[i][r]  # 增加一跳服务请求计数
                    total_latency += requests[i][r] * (beta * nearest_hop + alpha)  # 增加总延迟
                    final_traffic += requests[i][r] * (beta * nearest_hop + gamma)  # 增加最终流量
                else:
                    cloud_served_requests += requests[i][r]  # 增加云端服务请求计数
                    total_latency += requests[i][r] * gamma  # 增加总延迟
                    final_traffic += requests[i][r] * gamma  # 增加最终流量
    
    latency_reduction = ((initial_latency - total_latency) / initial_latency) * 100 if initial_latency > 0 else 0  # 计算延迟减少百分比
    traffic_reduction = ((initial_traffic - final_traffic) / initial_traffic) * 100 if initial_traffic > 0 else 0  # 计算流量减少百分比
    cloud_workload_reduction = ((initial_cloud_workload - cloud_served_requests) / initial_cloud_workload) * 100 if initial_cloud_workload > 0 else 0  # 计算云工作负载减少百分比
    
    throughput = total_requests / total_latency if total_latency > 0 else 0  # 计算吞吐率
    average_latency = total_latency / total_requests if total_requests > 0 else 0  # 计算平均延迟
    load = total_requests / N  # 计算负载
    
    return latency_reduction, traffic_reduction, cloud_workload_reduction, throughput, average_latency, load

def run_simulation(alpha, beta, gamma):
    requests = generate_requests_matrix(N, M)  # 生成用户请求矩阵
    G = create_network_topology(N)  # 创建网络拓扑
    cache_strategy = initialize_cache_strategy(N, M, p)  # 初始化缓存策略
    final_cache_strategy = update_cache_strategy(cache_strategy, requests, G, alpha, beta, gamma)  # 更新缓存策略
    utilities = [calculate_utility(i, final_cache_strategy, requests, G, alpha, beta, gamma) for i in range(N)]  # 计算效用
    performance_metrics = evaluate_performance(final_cache_strategy, requests, G, alpha, beta, gamma)  # 评估性能
    return utilities, performance_metrics, final_cache_strategy  # 返回结果

def main():
    parameter_sets = [
        {'alpha': 6.0, 'beta': 1.5, 'gamma': 9.0},
        {'alpha': 7.0, 'beta': 1.5, 'gamma': 9.0},
        {'alpha': 6.0, 'beta': 1.0, 'gamma': 9.0},
        {'alpha': 6.0, 'beta': 1.5, 'gamma': 8.0}
    ]
    
    results = []
    
    for params in parameter_sets:
        alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
        print(f"Running simulation with parameters: alpha={alpha}, beta={beta}, gamma={gamma}")  # 输出参数
        utilities, performance_metrics, final_cache_strategy = run_simulation(alpha, beta, gamma)  # 运行仿真
        results.append({
            'parameters': params,
            'utilities': utilities,
            'performance_metrics': performance_metrics,
            'final_cache_strategy': final_cache_strategy
        })
        print("Utilities of Base Stations:", utilities)  # 输出效用
        print("Performance Metrics:", performance_metrics)  # 输出性能指标
        print("-" * 80)  # 分隔线
    
    plt.figure(figsize=(16, len(parameter_sets) * 6))  # 设置图形大小
    
    for idx, result in enumerate(results):
        params = result['parameters']
        final_cache_strategy = result['final_cache_strategy']
        utilities = result['utilities']
        performance_metrics = result['performance_metrics']
        
        plt.subplot(len(parameter_sets), 3, idx * 3 + 1)  # 绘制缓存策略
        plt.imshow(final_cache_strategy, cmap='binary', aspect='auto')
        plt.title(f'Cache Strategy (alpha={params["alpha"]}, beta={params["beta"]}, gamma={params["gamma"]})')  # 标题
        plt.xlabel('Video ID')  # X轴标签
        plt.ylabel('Base Station ID')  # Y轴标签
        
        plt.subplot(len(parameter_sets), 3, idx * 3 + 2)  # 绘制效用
        plt.bar(range(N), utilities)
        plt.title(f'Utilities of Base Stations (alpha={params["alpha"]}, beta={params["beta"]}, gamma={params["gamma"]})')  # 标题
        plt.xlabel('Base Station ID')  # X轴标签
        plt.ylabel('Utility')  # Y轴标签
        
        plt.subplot(len(parameter_sets), 3, idx * 3 + 3)  # 绘制性能指标
        metrics_labels = ['Latency Reduction', 'Traffic Reduction', 'Cloud Workload Reduction']
        metrics_values = performance_metrics[:3]
        plt.bar(metrics_labels, metrics_values, color=['blue', 'green', 'red'])
        plt.title(f'Performance Metrics (alpha={params["alpha"]}, beta={params["beta"]}, gamma={params["gamma"]})')  # 标题
        plt.ylim(0, 100)  # Y轴范围
        plt.ylabel('% Reduction')  # Y轴标签
    
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形

if __name__ == "__main__":
    main()




