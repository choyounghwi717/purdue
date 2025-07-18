import streamlit as st
import pandas as pd
import numpy as np

# ✅ 고정된 CSV 데이터 로드
files = [
    "FOOD-DATA-GROUP1.csv",
    "FOOD-DATA-GROUP2.csv",
    "FOOD-DATA-GROUP3.csv",
    "FOOD-DATA-GROUP4.csv",
    "FOOD-DATA-GROUP5.csv"
]
food_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
food_data.drop_duplicates(inplace=True)
ALL_NUTRIENTS = [col for col in food_data.columns if col.lower() not in ['name', 'id', 'food']]

# ✅ 세션 상태 초기화
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'max_foods' not in st.session_state:
    st.session_state.max_foods = 10
if 'selected_nutrients' not in st.session_state:
    st.session_state.selected_nutrients = []
if 'constraints' not in st.session_state:
    st.session_state.constraints = {
        'required_ranges': {},
        'upper_limits': {},
        'soft_targets': {}
    }
if 'fixed_food_name' not in st.session_state:
    st.session_state.fixed_food_name = None

# ✅ STEP 1: 음식 개수 입력
if st.session_state.step == 1:
    st.header("1️⃣ 총 음식 개수를 입력하세요")
    max_foods = st.number_input("최대 음식 개수", min_value=1, max_value=50, value=10, step=1)
    if st.button("다음"):
        st.session_state.max_foods = max_foods
        st.session_state.step = 2
        st.experimental_rerun()

# ✅ STEP 2: 고정 음식 선택
elif st.session_state.step == 2:
    st.header("2️⃣ 반드시 포함할 음식을 선택하세요 (선택하지 않아도 됩니다)")
    search = st.text_input("음식 이름을 검색하세요")
    matches = food_data[food_data['name'].str.contains(search, case=False)] if search else pd.DataFrame()
    selected_food = None
    if not matches.empty:
        selected_food = st.selectbox("음식을 선택하세요", matches['name'].tolist())
    if st.button("다음", key="to_step3"):
        if selected_food:
            st.session_state.fixed_food_name = selected_food
        else:
            st.session_state.fixed_food_name = None
        st.session_state.step = 3
        st.experimental_rerun()
    if st.button("이전"):
        st.session_state.step = 1
        st.experimental_rerun()

# ✅ STEP 3: 영양소 선택
elif st.session_state.step == 3:
    st.header("3️⃣ 신경 써야 할 영양소를 선택하세요")
    selected = st.multiselect("영양소 목록", ALL_NUTRIENTS)
    if st.button("다음", key="to_step4"):
        st.session_state.selected_nutrients = selected
        st.session_state.step = 4
        st.experimental_rerun()
    if st.button("이전", key="back_to_step2"):
        st.session_state.step = 2
        st.experimental_rerun()

# ✅ STEP 4: 영양소 분류 및 값 입력
elif st.session_state.step == 4:
    st.header("4️⃣ 선택한 영양소를 분류하고 값을 입력하세요")
    for nutrient in st.session_state.selected_nutrients:
        category = st.radio(
            f"\U0001F9EC '{nutrient}'을(를) 어떻게 다룰까요?",
            options=["required_ranges", "upper_limits", "soft_targets"],
            horizontal=True,
            key=f"{nutrient}_type"
        )
        if category == "required_ranges":
            min_val = st.number_input(f"{nutrient} 최소값", key=f"{nutrient}_min")
            max_val = st.number_input(f"{nutrient} 최대값 (0이면 무제한)", key=f"{nutrient}_max")
            max_val = None if max_val == 0 else max_val
            st.session_state.constraints['required_ranges'][nutrient] = (min_val, max_val)
        elif category == "upper_limits":
            max_val = st.number_input(f"{nutrient} 상한값", key=f"{nutrient}_upper")
            st.session_state.constraints['upper_limits'][nutrient] = max_val
        elif category == "soft_targets":
            goal_val = st.number_input(f"{nutrient} 목표값", key=f"{nutrient}_soft")
            st.session_state.constraints['soft_targets'][nutrient] = goal_val
    if st.button("제약 조건 확인"):
        st.subheader("✅ 설정된 제약 조건")
        st.json(st.session_state.constraints)
    if st.button("이전", key="back_to_step3"):
        st.session_state.step = 3
        st.experimental_rerun()

# ✅ GA 알고리즘 정의 및 실행

def compute_penalty(ind, data, constraints, max_foods=10):
    selected = data[ind == 1]
    totals = selected.sum()
    penalty = 0
    for nutrient, (low, high) in constraints['required_ranges'].items():
        if nutrient in selected.columns:
            value = totals[nutrient]
            if low and value < low:
                penalty += (low - value) ** 2
            if high and value > high:
                penalty += (value - high) ** 2
    for nutrient, limit in constraints['upper_limits'].items():
        if nutrient in selected.columns:
            value = totals[nutrient]
            if value > limit:
                penalty += (value - limit) ** 2
    for nutrient, target in constraints['soft_targets'].items():
        if nutrient in selected.columns:
            value = totals[nutrient]
            penalty += 0.01 * (value - target) ** 2
    if np.sum(ind) > max_foods:
        penalty += ((np.sum(ind) - max_foods) ** 2) * 1000
    return min(penalty, 1e9)

def enforce_food_limit(ind, max_foods=10):
    ones = np.where(ind == 1)[0]
    if len(ones) > max_foods:
        remove = np.random.choice(ones, len(ones) - max_foods, replace=False)
        ind[remove] = 0
    return ind

def enforce_food_limit_with_fixed(ind, fixed_index, max_foods=10):
    ind[fixed_index] = 1
    ones = np.where(ind == 1)[0]
    if len(ones) > max_foods:
        removable = [i for i in ones if i != fixed_index]
        remove = np.random.choice(removable, len(ones) - max_foods, replace=False)
        ind[remove] = 0
    return ind

def initialize_population(pop_size, num_features, max_foods=10):
    pop = []
    for _ in range(pop_size):
        ind = np.zeros(num_features, dtype=int)
        selected = np.random.choice(range(num_features), max_foods, replace=False)
        ind[selected] = 1
        pop.append(ind)
    return pop

def initialize_population_fixed(pop_size, num_features, fixed_index, max_foods=10):
    pop = []
    for _ in range(pop_size):
        ind = np.zeros(num_features, dtype=int)
        ind[fixed_index] = 1
        others = [i for i in range(num_features) if i != fixed_index]
        selected = np.random.choice(others, max_foods - 1, replace=False)
        ind[selected] = 1
        pop.append(ind)
    return pop

def mutate(ind, prob=0.05, max_foods=10):
    for i in range(len(ind)):
        if np.random.rand() < prob:
            ind[i] = 1 - ind[i]
    return enforce_food_limit(ind, max_foods)

def mutate_fixed(ind, fixed_index, prob=0.05, max_foods=10):
    for i in range(len(ind)):
        if i != fixed_index and np.random.rand() < prob:
            ind[i] = 1 - ind[i]
    return enforce_food_limit_with_fixed(ind, fixed_index, max_foods)

def crossover(parent1, parent2, prob=0.8, max_foods=10):
    if np.random.rand() < prob:
        point = np.random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    child1 = enforce_food_limit(child1, max_foods)
    child2 = enforce_food_limit(child2, max_foods)
    return child1, child2

def crossover_fixed(parent1, parent2, fixed_index, prob=0.8, max_foods=10):
    if np.random.rand() < prob:
        point = np.random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    child1 = enforce_food_limit_with_fixed(child1, fixed_index, max_foods)
    child2 = enforce_food_limit_with_fixed(child2, fixed_index, max_foods)
    return child1, child2

def run_ga(data, constraints, pop_size=100, generations=50, max_foods=10, fixed_index=None):
    num_features = data.shape[0]
    if fixed_index is not None:
        population = initialize_population_fixed(pop_size, num_features, fixed_index, max_foods)
    else:
        population = initialize_population(pop_size, num_features, max_foods)
    best_penalty = float('inf')
    for gen in range(generations):
        penalties = [compute_penalty(ind, data, constraints, max_foods) for ind in population]
        new_population = []
        for _ in range(pop_size // 2):
            idxs = np.random.choice(len(population), 2, replace=False)
            p1, p2 = population[idxs[0]], population[idxs[1]]
            if fixed_index is not None:
                c1, c2 = crossover_fixed(p1, p2, fixed_index, max_foods=max_foods)
                c1 = mutate_fixed(c1, fixed_index, max_foods=max_foods)
                c2 = mutate_fixed(c2, fixed_index, max_foods=max_foods)
            else:
                c1, c2 = crossover(p1, p2, max_foods=max_foods)
                c1 = mutate(c1, max_foods=max_foods)
                c2 = mutate(c2, max_foods=max_foods)
            new_population.extend([c1, c2])
        population = new_population
        gen_best_penalty = min([compute_penalty(ind, data, constraints, max_foods) for ind in population])
        best_penalty = min(best_penalty, gen_best_penalty)
    final_penalties = [compute_penalty(ind, data, constraints, max_foods) for ind in population]
    best_idx = np.argmin(final_penalties)
    best_individual = population[best_idx]
    return best_individual, best_penalty

# ✅ 실행 버튼
if st.button("✅ 식단 추천 실행"):
    st.info("유전 알고리즘 실행 중입니다... ⏳")
    fixed_index = None
    if st.session_state.fixed_food_name:
        fixed_index = food_data[food_data['name'] == st.session_state.fixed_food_name].index[0]
    best_ind, best_penalty = run_ga(
        food_data,
        st.session_state.constraints,
        pop_size=100,
        generations=50,
        max_foods=st.session_state.max_foods,
        fixed_index=fixed_index
    )
    selected_foods = food_data[best_ind == 1]
    totals = selected_foods.sum()
    st.success(f"🎯 Best Penalty: {best_penalty:.2f}")
    st.subheader("🥗 추천 식단 (선택된 음식들):")
    st.dataframe(selected_foods)
    st.subheader("📊 총합 영양소:")
    st.dataframe(totals.to_frame(name="합계"))
    csv = selected_foods.to_csv(index=False).encode('utf-8')
    st.download_button("📥 CSV로 저장", csv, "recommended_diet.csv", "text/csv")
