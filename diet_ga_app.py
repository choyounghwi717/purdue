import streamlit as st

# 전체 영양소 목록 예시 (원하는 대로 수정 가능)
ALL_NUTRIENTS = [
    'Caloric Value', 'Protein', 'Calcium', 'Vitamin D', 'Magnesium',
    'Polyunsaturated Fats', 'Saturated Fats', 'Sodium', 'Sugars', 'Cholesterol',
    'Potassium', 'Vitamin C', 'Vitamin E', 'Vitamin K', 'Copper', 'Iron', 'Nutrition Density'
]

# 세션 상태 초기화
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

# --- STEP 1: 음식 개수 입력 ---
if st.session_state.step == 1:
    st.header("1️⃣ 총 음식 개수를 입력하세요")
    max_foods = st.number_input("최대 음식 개수", min_value=1, max_value=50, value=10, step=1)

    if st.button("다음"):
        st.session_state.max_foods = max_foods
        st.session_state.step = 2
        st.experimental_rerun()

# --- STEP 2: 신경쓸 영양소 선택 ---
elif st.session_state.step == 2:
    st.header("2️⃣ 신경써야 할 영양소를 선택하세요")
    selected = st.multiselect("영양소 목록", ALL_NUTRIENTS)

    if st.button("다음", key="to_step3"):
        st.session_state.selected_nutrients = selected
        st.session_state.step = 3
        st.experimental_rerun()
        

    if st.button("이전"):
        st.session_state.step = 1
        st.experimental_rerun()

# --- STEP 3: 각 영양소 분류 및 값 입력 ---
elif st.session_state.step == 3:
    st.header("3️⃣ 선택된 영양소를 분류하고 값을 입력하세요")

    for nutrient in st.session_state.selected_nutrients:
        category = st.radio(
            f"🧬 '{nutrient}'을(를) 어떻게 다룰까요?",
            options=["required_ranges", "upper_limits", "soft_targets"],
            horizontal=True,
            key=f"{nutrient}_type"
        )

        if category == "required_ranges":
            min_val = st.number_input(f"{nutrient} 최소값", key=f"{nutrient}_min")
            max_val = st.number_input(f"{nutrient} 최대값 (None이면 비워두세요)", key=f"{nutrient}_max")
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

    if st.button("이전", key="to_step2"):
        st.session_state.step = 2
        st.experimental_rerun()
