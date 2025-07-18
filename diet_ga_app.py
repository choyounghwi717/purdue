import streamlit as st
import pandas as pd
import numpy as np

# ✅ 데이터 로드
FILES = [
    "FOOD-DATA-GROUP1.csv",
    "FOOD-DATA-GROUP2.csv",
    "FOOD-DATA-GROUP3.csv",
    "FOOD-DATA-GROUP4.csv",
    "FOOD-DATA-GROUP5.csv"
]
dfs = [pd.read_csv(f) for f in FILES]
food_data = pd.concat(dfs, ignore_index=True).drop_duplicates()
ALL_NUTRIENTS = [col for col in food_data.columns if col.lower() not in ['id', 'food']]

# ✅ 세션 상태 초기화
if "step" not in st.session_state:
    st.session_state.step = 0
if "max_foods" not in st.session_state:
    st.session_state.max_foods = 10
if "fixed_food_name" not in st.session_state:
    st.session_state.fixed_food_name = ""
if "nutrient_types" not in st.session_state:
    st.session_state.nutrient_types = {}
if "constraints" not in st.session_state:
    st.session_state.constraints = {}
if "no_fixed_food" not in st.session_state:
    st.session_state.no_fixed_food = False

# ✅ Step 0: 음식 개수 입력
if st.session_state.step == 0:
    st.header("1️⃣ 총 음식 개수를 입력하세요")
    count = st.number_input("최대 음식 개수", 5, 30, 10)
    if st.button("다음"):
        st.session_state.max_foods = count
        st.session_state.step = 1

# ✅ Step 1: 고정 음식 입력 또는 없음을 선택
elif st.session_state.step == 1:
    st.header("2️⃣ 포함할 음식이 있다면 검색하세요")
    name = st.text_input("음식 이름 입력")
    no_food = st.checkbox("고정할 음식이 없습니다")
    if st.button("다음"):
        if not no_food and name in food_data['food'].values:
            st.session_state.fixed_food_name = name
            st.session_state.no_fixed_food = False
        else:
            st.session_state.fixed_food_name = ""
            st.session_state.no_fixed_food = True
        st.session_state.step = 2

# ✅ Step 2: 영양소 선택
elif st.session_state.step == 2:
    st.header("3️⃣ 고려할 영양소를 선택하세요")
    selected = st.multiselect("영양소 선택", ALL_NUTRIENTS)
    if st.button("다음"):
        if selected:
            st.session_state.selected_nutrients = selected
            st.session_state.step = 3

# ✅ Step 3: 각 영양소의 제약 유형 설정
elif st.session_state.step == 3:
    st.header("4️⃣ 제약 유형을 하나씩 지정하세요")
    nutrient_types = {}
    for nutrient in st.session_state.selected_nutrients:
        ctype = st.selectbox(
            f"{nutrient}의 제약 유형",
            ["범위", "상한", "권장"],
            key=f"type_{nutrient}"
        )
        nutrient_types[nutrient] = ctype
    if st.button("다음"):
        st.session_state.nutrient_types = nutrient_types
        st.session_state.step = 4

# ✅ Step 4: 각 유형에 맞는 값 입력
elif st.session_state.step == 4:
    st.header("5️⃣ 제약 조건 값 입력")
    required_ranges, upper_limits, soft_targets = {}, {}, {}
    for nutrient, ctype in st.session_state.nutrient_types.items():
        if ctype == "범위":
            low = st.number_input(f"{nutrient} 최소값", key=f"min_{nutrient}", value=0.0)
            high = st.number_input(f"{nutrient} 최대값", key=f"max_{nutrient}", value=0.0)
            high = None if high == 0 else high
            required_ranges[nutrient] = (low, high)
        elif ctype == "상한":
            limit = st.number_input(f"{nutrient} 상한", key=f"upper_{nutrient}", value=0.0)
            if limit > 0:
                upper_limits[nutrient] = limit
        elif ctype == "권장":
            goal = st.number_input(f"{nutrient} 권장값", key=f"soft_{nutrient}", value=0.0)
            if goal > 0:
                soft_targets[nutrient] = goal
    if st.button("식단 추천 실행"):
        st.session_state.constraints = {
            'required_ranges': required_ranges,
            'upper_limits': upper_limits,
            'soft_targets': soft_targets
        }
        st.session_state.step = 5

# ✅ Step 5: 결과 출력
elif st.session_state.step == 5:
    st.header("✅ 추천 식단 결과")

    def enforce(ind, fixed_idx, maxf):
        if fixed_idx is not None:
            ind[fixed_idx] = 1
        ones = np.where(ind == 1)[0]
        if len(ones) > maxf:
            remove = [i for i in ones if i != fixed_idx]
            drop = np.random.choice(remove, len(ones) - maxf, replace=False)
            ind[drop] = 0
        return ind

    def initialize(pop_size, dim, fixed_idx, maxf):
        pop = []
        for _ in range(pop_size):
            ind = np.zeros(dim, dtype=int)
            if fixed_idx is not None:
                ind[fixed_idx] = 1
            others = [i for i in range(dim) if i != fixed_idx]
            sel = np.random.choice(others, maxf - (1 if fixed_idx is not None else 0), replace=False)
            ind[sel] = 1
            pop.append(ind)
        return pop

    def mutate(ind, fixed_idx, maxf, prob=0.05):
        for i in range(len(ind)):
            if i != fixed_idx and np.random.rand() < prob:
                ind[i] = 1 - ind[i]
        return enforce(ind, fixed_idx, maxf)

    def crossover(p1, p2, fixed_idx, maxf, prob=0.8):
        if np.random.rand() < prob:
            pt = np.random.randint(1, len(p1)-1)
            c1 = np.concatenate([p1[:pt], p2[pt:]])
            c2 = np.concatenate([p2[:pt], p1[pt:]])
        else:
            c1, c2 = p1.copy(), p2.copy()
        return enforce(c1, fixed_idx, maxf), enforce(c2, fixed_idx, maxf)

    def penalty(ind, data, cons, maxf):
        sel = data[ind == 1]
        total = sel.sum()
        p = 0
        for n, (low, high) in cons['required_ranges'].items():
            if n in total:
                v = total[n]
                if low and v < low:
                    p += (low - v) ** 2
                if high and v > high:
                    p += (v - high) ** 2
        for n, lim in cons['upper_limits'].items():
            if n in total and total[n] > lim:
                p += (total[n] - lim) ** 2
        for n, tgt in cons['soft_targets'].items():
            if n in total:
                p += 0.01 * (total[n] - tgt) ** 2
        if np.sum(ind) > maxf:
            p += 1000 * (np.sum(ind) - maxf) ** 2
        return min(p, 1e9)

    def run_ga(data, cons, fixed_idx, maxf, gen=50, pop_size=100):
        dim = data.shape[0]
        pop = initialize(pop_size, dim, fixed_idx, maxf)
        for _ in range(gen):
            scores = [penalty(ind, data, cons, maxf) for ind in pop]
            new_pop = []
            for _ in range(pop_size // 2):
                i1, i2 = np.random.choice(len(pop), 2, replace=False)
                c1, c2 = crossover(pop[i1], pop[i2], fixed_idx, maxf)
                c1 = mutate(c1, fixed_idx, maxf)
                c2 = mutate(c2, fixed_idx, maxf)
                new_pop.extend([c1, c2])
            pop = new_pop
        best = min(pop, key=lambda x: penalty(x, data, cons, maxf))
        return best

    fixed_idx = None
    if not st.session_state.no_fixed_food and st.session_state.fixed_food_name in food_data['food'].values:
        fixed_idx = food_data[food_data['food'] == st.session_state.fixed_food_name].index[0]

    best = run_ga(food_data, st.session_state.constraints, fixed_idx, st.session_state.max_foods)

    selected = food_data[best == 1]

    st.subheader("🍱 추천 식단 (이름만 표시)")
    st.dataframe(selected[['food']])

    st.subheader("📊 총합 영양소 (전체)")
    nutrient_total = selected.drop(columns=['id', 'food'], errors='ignore').sum().to_frame("합계")
    st.dataframe(nutrient_total)

    csv = selected.to_csv(index=False).encode('utf-8')
    st.download_button("📥 추천 식단 CSV 다운로드", csv, "recommended_diet.csv", "text/csv")
