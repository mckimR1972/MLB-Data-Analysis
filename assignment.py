import time
import sys
import heapq

sys.setrecursionlimit(1000000)

# Brute Force (DFS 기반 + 2^n개 부분집합 모두 탐색)
def brute_force(n: int, W: int, weights: list, values: list) -> dict:
    stats = {'visited': 0, 'best': 0}

    def dfs(level: int, cur_weight: int, cur_value: int):
        stats['visited'] += 1

        if level == n:
            if cur_weight <= W and cur_value > stats['best']:
                stats['best'] = cur_value
            return

        dfs(level + 1, cur_weight + weights[level], cur_value + values[level])
        dfs(level + 1, cur_weight, cur_value)

    dfs(0, 0, 0)
    return {'best': stats['best'], 'visited': stats['visited'], 'pruned': 0}


# Dynamic Programming, dp[i][w] = 물건 1~i까지 고려, 용량 w일 때 최대 가치
def dynamic_programming(n: int, W: int, weights: list, values: list) -> dict:
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i = weights[i - 1]
        v_i = values[i - 1]
        for w in range(W + 1):
            if w_i <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)
            else:
                dp[i][w] = dp[i - 1][w]

    return {'best': dp[n][W], 'visited': n * (W + 1), 'pruned': 0}


# 느슨한 상한의 Branch and Bound(Best-First Search)
# 상한 = 현재 가치 + 남은 모든 물건의 가치 합
def loose_upper_bound(level: int, cur_value: int, n: int, values: list) -> float:
    ub = cur_value
    for i in range(level, n):
        ub += values[i]
    return ub


def bnb_loose(n: int, W: int, weights: list, values: list) -> dict:
    stats = {'visited': 0, 'pruned': 0, 'best': 0}

    init_ub = loose_upper_bound(0, 0, n, values)
    pq = [(-init_ub, 0, 0, 0)]

    while pq:
        neg_ub, level, cur_weight, cur_value = heapq.heappop(pq)
        ub = -neg_ub
        stats['visited'] += 1

        if ub <= stats['best']:
            stats['pruned'] += 1
            continue

        if level == n:
            if cur_value > stats['best']:
                stats['best'] = cur_value
            continue

        new_weight = cur_weight + weights[level]
        new_value = cur_value + values[level]
        if new_weight <= W:
            child_ub = loose_upper_bound(level + 1, new_value, n, values)
            if child_ub > stats['best']:
                heapq.heappush(pq, (-child_ub, level + 1, new_weight, new_value))
            else:
                stats['pruned'] += 1
        else:
            stats['pruned'] += 1

        child_ub = loose_upper_bound(level + 1, cur_value, n, values)
        if child_ub > stats['best']:
            heapq.heappush(pq, (-child_ub, level + 1, cur_weight, cur_value))
        else:
            stats['pruned'] += 1

    return {'best': stats['best'], 'visited': stats['visited'], 'pruned': stats['pruned']}


# 4. Branch and Bound - Fractional Knapsack 상한 (Best-First Search)
# 상한 = 현재 가치 + 남은 용량에 대해 단위가치 높은 순으로
def fractional_upper_bound(level: int, cur_weight: int, cur_value: int,
                           n: int, W: int, sorted_w: list, sorted_v: list) -> float:
    ub = cur_value
    remaining_cap = W - cur_weight

    for i in range(level, n):
        if sorted_w[i] <= remaining_cap:
            ub += sorted_v[i]
            remaining_cap -= sorted_w[i]
        else:
            ub += sorted_v[i] * (remaining_cap / sorted_w[i])
            break

    return ub


def bnb_fractional(n: int, W: int, weights: list, values: list) -> dict:
    stats = {'visited': 0, 'pruned': 0, 'best': 0}

    
    order = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    sorted_w = [weights[order[i]] for i in range(n)]
    sorted_v = [values[order[i]] for i in range(n)]

    init_ub = fractional_upper_bound(0, 0, 0, n, W, sorted_w, sorted_v)
    
    pq = [(-init_ub, 0, 0, 0)]

    while pq:
        neg_ub, level, cur_weight, cur_value = heapq.heappop(pq)
        ub = -neg_ub
        stats['visited'] += 1

        if ub <= stats['best']:
            stats['pruned'] += 1
            continue

        if level == n:
            if cur_value > stats['best']:
                stats['best'] = cur_value
            continue

        new_weight = cur_weight + sorted_w[level]
        new_value = cur_value + sorted_v[level]
        if new_weight <= W:
            child_ub = fractional_upper_bound(level + 1, new_weight, new_value,
                                              n, W, sorted_w, sorted_v)
            if child_ub > stats['best']:
                heapq.heappush(pq, (-child_ub, level + 1, new_weight, new_value))
            else:
                stats['pruned'] += 1
        else:
            stats['pruned'] += 1

        child_ub = fractional_upper_bound(level + 1, cur_weight, cur_value,
                                          n, W, sorted_w, sorted_v)
        if child_ub > stats['best']:
            heapq.heappush(pq, (-child_ub, level + 1, cur_weight, cur_value))
        else:
            stats['pruned'] += 1

    return {'best': stats['best'], 'visited': stats['visited'], 'pruned': stats['pruned']}

def fmt_time(elapsed):
    if elapsed >= 1:
        return f"{elapsed:.4f}초"
    elif elapsed >= 0.001:
        return f"{elapsed * 1000:.4f}ms"
    else:
        return f"{elapsed * 1_000_000:.2f}μs"

if __name__ == "__main__":
    filenames = ['knapsack_small.txt', 'knapsack_large_w.txt',
                 'knapsack_bnb_easy.txt', 'knapsack_bnb_hard.txt']

    all_results = {}

    for filename in filenames:
        print(f"\n{'='*60}")
        print(f"테스트 파일: {filename}")
        print('='*60)

        try:
            with open(filename, 'r') as f:
                first_line = f.readline().split()
                n, W = int(first_line[0]), int(first_line[1])
                weights = []
                values = []
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    weights.append(int(parts[0]))
                    values.append(int(parts[1]))
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {filename}")
            continue

        print(f"물건 수: {n}, 배낭 용량: {W}\n")

        file_results = {}

        # 1) Brute Force
        print(f"[Brute Force]")
        start = time.perf_counter()
        result_bf = brute_force(n, W, weights, values)
        elapsed = time.perf_counter() - start

        print(f"최적 가치: {result_bf['best']}")
        print(f"실행 시간: {fmt_time(elapsed)}")
        print(f"방문 노드 수: {result_bf['visited']:,}")
        print()
        file_results['Brute Force'] = {**result_bf, 'time': elapsed}

        # 2) Dynamic Programming
        print(f"[Dynamic Programming]")
        start = time.perf_counter()
        result_dp = dynamic_programming(n, W, weights, values)
        elapsed = time.perf_counter() - start

        print(f"최적 가치: {result_dp['best']}")
        print(f"실행 시간: {fmt_time(elapsed)}")
        print(f"DP 테이블 크기: {n} x {W+1} = {n * (W + 1):,} 셀")
        print()
        file_results['DP'] = {**result_dp, 'time': elapsed}

        # 3) Branch and Bound (느슨한 상한)
        print(f"[Branch and Bound - 느슨한 상한]")
        start = time.perf_counter()
        result_bnb_loose = bnb_loose(n, W, weights, values)
        elapsed = time.perf_counter() - start

        print(f"최적 가치: {result_bnb_loose['best']}")
        print(f"실행 시간: {fmt_time(elapsed)}")
        print(f"방문 노드 수: {result_bnb_loose['visited']:,}")
        print(f"가지치기 횟수: {result_bnb_loose['pruned']:,}")
        print()
        file_results['BnB Loose'] = {**result_bnb_loose, 'time': elapsed}

        # 4) Branch and Bound (Fractional Knapsack 상한)
        print(f"[Branch and Bound - Fractional Knapsack 상한]")
        start = time.perf_counter()
        result_bnb_frac = bnb_fractional(n, W, weights, values)
        elapsed = time.perf_counter() - start

        print(f"최적 가치: {result_bnb_frac['best']}")
        print(f"실행 시간: {fmt_time(elapsed)}")
        print(f"방문 노드 수: {result_bnb_frac['visited']:,}")
        print(f"가지치기 횟수: {result_bnb_frac['pruned']:,}")
        print()
        file_results['BnB Fractional'] = {**result_bnb_frac, 'time': elapsed}

        # ── 정답 검증 ──
        answers = set()
        for method, res in file_results.items():
            answers.add(res['best'])
        if len(answers) == 1:
            print(f"검증 완료: 모든 알고리즘의 최적 가치가 {answers.pop()}으로 일치")
        else:
            print(f"주의: 알고리즘 결과가 불일치")
            for method, res in file_results.items():
                print(f"  {method}: {res['best']}")

        all_results[filename] = file_results