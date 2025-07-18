import streamlit as st
import pandas as pd
import numpy as np

# ✅ 1. CSV 불러오기 (GitHub에 업로드된 상태여야 함)
FILES = [
    "FOOD-DATA-GROUP1.csv",
    "FOOD-DATA-GROUP2.csv",
    "FOOD-DATA-GROUP3.csv",
    "FOOD-DATA-GROUP4.csv",
    "FOOD-DATA-GROUP5.csv"
]

dataframes = [pd.read_csv(f) for f in FILES]
food_data = pd.concat(dataframes, ignore_index=True).drop_duplicates()
ALL_NUTRIENTS = [col for col in food_data.columns if col.lower() not in ['id', 'name', 'food']]

# ✅ 2. 세션 초기화
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'max_foods' not in st.session_state:
    st.session_state.max_foods = 10
if 'selected_nutrients' not in st.session_state:
    st.session_state.selected_nutrients = []
if 'constraints' not in st.session_state:
    st.session_state.constraints = {}
if 'fixed_food_name' not in st.session_state:
    st.session_state.fixed_food_name = ""

# ✅ 3. Step 0: 음식 개수 입력
if st.session_state.step == 0:
    st.header("1️⃣ 총 음식 개수를 입력하세요")
    max_count = st.number_input("최대 음식 개수", min_value=5, max_value=30, value=10)
    if st.button("다음"):
        st.session_state.max_foods = max_count
        st.session_state.step = 1
        st.experimental_rerun()

# ✅ 4. Step 1: 고정 음식 선택
elif st.session_state.step == 1:
    st.header("2️⃣ 이미 포함할 음식이 있다면 검색하여 선택하세요")
    fixed_name = st.text_input("고정할 음식 이름 (예: Burrito with Cheese 등)")
    if fixed_name.strip() != "" and fixed_name in food_data['name'].values:
        st.session_state.fixed_food_name = fixed_name
    else:
        st.session_state.fixed_food_name = ""
    if st.button("다음", key="next1"):
        st.session_state.step = 2
        st.experimental_rerun()

# ✅ 5. Step 2: 영양소 선택
elif st.session_state.step == 2:
    st.header("3️⃣ 신경 쓸 영양소를 선택하세요")
    selected = st.multiselect("영양소 선택", ALL_NUTRIENTS)
    if st.button("다음", key="next2") and selected:
        st.session_state.selected_nutrients = selected
        st.session_state.step = 3
        st.experimental_rerun()

# ✅ 6. Step 3: 영양소 제약 입력
elif st.session_state.step == 3:
    st.header("4️⃣ 영양소 제약 조건 설정")
    required_ranges = {}
    upper_limits = {}
    soft_targets = {}

    st.subheader("Required Ranges (범위 설정)")
    for n in st.session_state.selected_nutrients:
        low = st.number_input(f"{n} 최소값", key=f"min_{n}", value=0.0)
        high = st.number_input(f"{n} 최대값 (생략시 무제한)", key=f"max_{n}", value=0.0)
        if high == 0:
            high = None
        required_ranges[n] = (low, high)

    st.subheader("Upper Limits (상한만 설정할 경우)")
    for n in st.session_state.selected_nutrients:
        limit = st.number_input(f"{n} 상한", key=f"upper_{n}", value=0.0)
        if limit > 0:
            upper_limits[n] = limit

    st.subheader("Soft Targets (권장값)")
    for n in st.session_state.selected_nutrients:
        goal = st.number_input(f"{n} 권장값", key=f"soft_{n}", value=0.0)
        if goal > 0:
            soft_targets[n] = goal

    if st.button("식단 추천 실행"):
        st.session_state.constraints = {
            'required_ranges': required_ranges,
            'upper_limits': upper_limits,
            'soft_targets': soft_targets
        }
        st.session_state.step = 4
        st.experimental_rerun()

# ✅ 7. Step 4: GA 실행
elif st.session_state.step == 4:
    st.header("✅ 추천 식단 결과")

    # GA 함수들
    def enforce_food_limit_with_fixed(ind, fixed_index, max_foods=10):
        if fixed_index is not None:
            ind[fixed_index] = 1
        ones = np.where(ind == 1)[0]
        if len(ones) > max_foods:
            removable = [i for i in ones if i != fixed_index] if fixed_index is not None else ones
            remove = np.random.choice(removable, len(ones) - max_foods, replace=False)
            ind[remove] = 0
        return ind

    def initialize_population(pop_size, num_features, fixed_index, max_foods=10):
        pop = []
        for _ in range(pop_size):
            ind = np.zeros(num_features, dtype=int)
            others = list(range(num_features))
            if fixed_index is not None:
                ind[fixed_index] = 1
                others.remove(fixed_index)
            selected = np.random.choice(others, max_foods - (1 if fixed_index is not None else 0), replace=False)
            ind[selected] = 1
            pop.append(ind)
        return pop

    def mutate(ind, fixed_index, prob=0.05, max_foods=10):
        for i in range(len(ind)):
            if i != fixed_index and np.random.rand() < prob:
                ind[i] = 1 - ind[i]
        return enforce_food_limit_with_fixed(ind, fixed_index, max_foods)

    def crossover(parent1, parent2, fixed_index, prob=0.8, max_foods=10):
        if np.random.rand() < prob:
            point = np.random.randint(1, len(parent1)-1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        return enforce_food_limit_with_fixed(child1, fixed_index, max_foods), enforce_food_limit_with_fixed(child2, fixed_index, max_foods)

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

    def run_ga(data, constraints, fixed_index=None, pop_size=100, generations=50, max_foods=10):
        num_features = data.shape[0]
        population = initialize_population(pop_size, num_features, fixed_index, max_foods)
        best_penalty = float('inf')

        for gen in range(generations):
            penalties = [compute_penalty(ind, data, constraints, max_foods) for ind in population]
            new_population = []
            for _ in range(pop_size // 2):
                idxs = np.random.choice(len(population), 2, replace=False)
                p1, p2 = population[idxs[0]], population[idxs[1]]
                c1, c2 = crossover(p1, p2, fixed_index, max_foods=max_foods)
                c1 = mutate(c1, fixed_index, max_foods=max_foods)
                c2 = mutate(c2, fixed_index, max_foods=max_foods)
                new_population.extend([c1, c2])
            population = new_population

        final_penalties = [compute_penalty(ind, data, constraints, max_foods) for ind in population]
        best_idx = np.argmin(final_penalties)
        return population[best_idx], final_penalties[best_idx]

    # 실행
    fixed_index = None
    if st.session_state.fixed_food_name and st.session_state.fixed_food_name in food_data['name'].values:
        fixed_index = food_data[food_data['name'] == st.session_state.fixed_food_name].index[0]

    best_ind, best_penalty = run_ga(
        food_data,
        st.session_state.constraints,
        fixed_index=fixed_index,
        pop_size=100,
        generations=50,
        max_foods=st.session_state.max_foods
    )

    selected = food_data[best_ind == 1]
    valid_columns = [col for col in st.session_state.selected_nutrients if col in selected.columns]

    st.subheader("📋 추천 식단")
    st.dataframe(selected[['name'] + valid_columns])

    st.subheader("📊 총합 영양소")
    st.dataframe(selected[valid_columns].sum().to_frame("합계"))

    csv = selected.to_csv(index=False).encode('utf-8')
    st.download_button("📥 추천 식단 CSV 다운로드", csv, file_name="recommended_diet.csv", mime="text/csv")
