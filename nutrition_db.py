"""
FitAI - nutrition_db.py
========================
Static nutrition lookup table — mess + street food items.
All values: per 100g. portion_g = standard single serving in grams.
serving_desc = human-readable portion description shown to user.
Sources: IFCT 2017, NIN Hyderabad, standard recipe estimates.
"""

NUTRITION_DB: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════════
    # MESS — BREAKFAST
    # ══════════════════════════════════════════════════════════════════════════

    "dahi_vada": {
        "calories": 180, "protein": 6.5, "carbs": 28.0, "fats": 5.0,
        "portion_g": 150, "serving_desc": "2 pieces with dahi",
    },
    "aloo_dum": {
        "calories": 110, "protein": 2.2, "carbs": 18.0, "fats": 3.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "onion_sev": {
        "calories": 420, "protein": 8.0, "carbs": 52.0, "fats": 20.0,
        "portion_g": 80, "serving_desc": "1 small plate",
    },
    "corn_flakes": {
        "calories": 357, "protein": 7.0, "carbs": 84.0, "fats": 0.4,
        "portion_g": 40, "serving_desc": "1 bowl (dry)",
    },
    "corn_flakes_with_milk_and_banana": {
        "calories": 145, "protein": 5.5, "carbs": 28.0, "fats": 2.0,
        "portion_g": 280, "serving_desc": "1 bowl with milk & banana",
    },
    "coffee": {
        "calories": 35, "protein": 1.2, "carbs": 4.5, "fats": 1.2,
        "portion_g": 150, "serving_desc": "1 cup",
    },
    "tea": {
        "calories": 30, "protein": 0.8, "carbs": 4.0, "fats": 0.8,
        "portion_g": 150, "serving_desc": "1 cup",
    },
    "maggi": {
        "calories": 385, "protein": 8.0, "carbs": 53.0, "fats": 16.0,
        "portion_g": 130, "serving_desc": "1 packet (cooked)",
    },
    "veg_maggi": {
        "calories": 210, "protein": 4.8, "carbs": 28.0, "fats": 9.0,
        "portion_g": 150, "serving_desc": "1 plate",
    },
    "egg_maggi": {
        "calories": 245, "protein": 10.0, "carbs": 27.0, "fats": 11.0,
        "portion_g": 160, "serving_desc": "1 plate with 1 egg",
    },
    "chicken_maggi": {
        "calories": 255, "protein": 12.0, "carbs": 27.0, "fats": 11.0,
        "portion_g": 160, "serving_desc": "1 plate with chicken",
    },
    "cheese_maggi": {
        "calories": 280, "protein": 10.0, "carbs": 27.0, "fats": 14.0,
        "portion_g": 150, "serving_desc": "1 plate with cheese",
    },
    "uttapam": {
        "calories": 170, "protein": 5.0, "carbs": 30.0, "fats": 3.5,
        "portion_g": 120, "serving_desc": "1 piece",
    },
    "sambar": {
        "calories": 55, "protein": 2.8, "carbs": 8.5, "fats": 1.2,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "chutney": {
        "calories": 60, "protein": 1.0, "carbs": 8.0, "fats": 2.5,
        "portion_g": 30, "serving_desc": "2 tablespoons",
    },
    "bread": {
        "calories": 265, "protein": 9.0, "carbs": 51.0, "fats": 3.2,
        "portion_g": 60, "serving_desc": "2 slices",
    },
    "butter": {
        "calories": 717, "protein": 0.9, "carbs": 0.1, "fats": 81.0,
        "portion_g": 10, "serving_desc": "1 teaspoon",
    },
    "jam": {
        "calories": 250, "protein": 0.4, "carbs": 62.0, "fats": 0.1,
        "portion_g": 20, "serving_desc": "1 tablespoon",
    },
    "cutlet": {
        "calories": 220, "protein": 8.0, "carbs": 22.0, "fats": 11.0,
        "portion_g": 100, "serving_desc": "1 piece",
    },
    "egg_omelette": {
        "calories": 185, "protein": 12.0, "carbs": 1.5, "fats": 14.0,
        "portion_g": 100, "serving_desc": "1 omelette (2 eggs)",
    },
    "omelette": {
        "calories": 185, "protein": 12.0, "carbs": 1.5, "fats": 14.0,
        "portion_g": 100, "serving_desc": "1 omelette (2 eggs)",
    },
    "cheese_omelette": {
        "calories": 230, "protein": 15.0, "carbs": 2.0, "fats": 18.0,
        "portion_g": 120, "serving_desc": "1 omelette with cheese",
    },
    "upma": {
        "calories": 130, "protein": 3.5, "carbs": 22.0, "fats": 3.5,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "ghugni": {
        "calories": 140, "protein": 7.5, "carbs": 22.0, "fats": 3.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "idli": {
        "calories": 58, "protein": 2.0, "carbs": 12.0, "fats": 0.4,
        "portion_g": 150, "serving_desc": "3 pieces",
    },
    "boiled_egg": {
        "calories": 155, "protein": 13.0, "carbs": 1.1, "fats": 11.0,
        "portion_g": 60, "serving_desc": "1 egg",
    },
    "vada": {
        "calories": 300, "protein": 7.0, "carbs": 35.0, "fats": 15.0,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "medu_vada": {
        "calories": 300, "protein": 7.0, "carbs": 35.0, "fats": 15.0,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "vada_sambhar": {
        "calories": 180, "protein": 6.0, "carbs": 22.0, "fats": 8.0,
        "portion_g": 230, "serving_desc": "1 vada with 1 bowl sambar",
    },
    "veg_chowmein": {
        "calories": 160, "protein": 4.5, "carbs": 28.0, "fats": 4.0,
        "portion_g": 250, "serving_desc": "1 plate",
    },
    "dosa": {
        "calories": 168, "protein": 3.8, "carbs": 30.0, "fats": 4.0,
        "portion_g": 100, "serving_desc": "1 piece",
    },
    "masala_dosa": {
        "calories": 175, "protein": 4.2, "carbs": 30.0, "fats": 5.5,
        "portion_g": 200, "serving_desc": "1 piece with filling",
    },
    "paneer_dosa": {
        "calories": 200, "protein": 8.0, "carbs": 28.0, "fats": 7.0,
        "portion_g": 180, "serving_desc": "1 piece with paneer filling",
    },
    "cheese_dosa": {
        "calories": 210, "protein": 7.0, "carbs": 28.0, "fats": 9.0,
        "portion_g": 180, "serving_desc": "1 piece with cheese",
    },
    "poha": {
        "calories": 110, "protein": 2.5, "carbs": 22.0, "fats": 2.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "rusk": {
        "calories": 400, "protein": 9.0, "carbs": 72.0, "fats": 9.0,
        "portion_g": 30, "serving_desc": "2 pieces",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MESS — STAPLES
    # ══════════════════════════════════════════════════════════════════════════

    "rice": {
        "calories": 130, "protein": 2.7, "carbs": 28.0, "fats": 0.3,
        "portion_g": 200, "serving_desc": "1 plate (medium)",
    },
    "roti": {
        "calories": 297, "protein": 8.0, "carbs": 57.0, "fats": 4.0,
        "portion_g": 40, "serving_desc": "1 piece",
    },
    "chapati": {
        "calories": 297, "protein": 8.0, "carbs": 57.0, "fats": 4.0,
        "portion_g": 40, "serving_desc": "1 piece",
    },
    "naan": {
        "calories": 310, "protein": 9.0, "carbs": 56.0, "fats": 6.0,
        "portion_g": 90, "serving_desc": "1 piece",
    },
    "puri": {
        "calories": 340, "protein": 6.0, "carbs": 45.0, "fats": 15.0,
        "portion_g": 35, "serving_desc": "1 piece",
    },
    "jeera_rice": {
        "calories": 148, "protein": 2.8, "carbs": 30.0, "fats": 2.5,
        "portion_g": 200, "serving_desc": "1 plate (medium)",
    },
    "khichdi": {
        "calories": 120, "protein": 4.5, "carbs": 22.0, "fats": 2.5,
        "portion_g": 250, "serving_desc": "1 bowl",
    },
    "sabudana_khichdi": {
        "calories": 200, "protein": 2.0, "carbs": 42.0, "fats": 4.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "veg_biryani": {
        "calories": 170, "protein": 4.0, "carbs": 32.0, "fats": 4.0,
        "portion_g": 300, "serving_desc": "1 plate",
    },
    "non_veg_biryani": {
        "calories": 190, "protein": 10.0, "carbs": 28.0, "fats": 5.5,
        "portion_g": 350, "serving_desc": "1 plate",
    },
    "chicken_biryani": {
        "calories": 190, "protein": 12.0, "carbs": 27.0, "fats": 5.5,
        "portion_g": 350, "serving_desc": "1 plate",
    },
    "egg_biryani": {
        "calories": 175, "protein": 8.0, "carbs": 28.0, "fats": 5.0,
        "portion_g": 300, "serving_desc": "1 plate",
    },
    "paneer_biryani": {
        "calories": 195, "protein": 8.0, "carbs": 28.0, "fats": 7.0,
        "portion_g": 300, "serving_desc": "1 plate",
    },
    "tawa_pulav": {
        "calories": 165, "protein": 4.0, "carbs": 30.0, "fats": 4.0,
        "portion_g": 250, "serving_desc": "1 plate",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MESS — DALS & CURRIES
    # ══════════════════════════════════════════════════════════════════════════

    "dal": {
        "calories": 116, "protein": 6.5, "carbs": 18.0, "fats": 3.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "dal_fry": {
        "calories": 120, "protein": 6.8, "carbs": 17.5, "fats": 3.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "dal_tadka": {
        "calories": 116, "protein": 6.5, "carbs": 18.0, "fats": 3.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "dal_makhani": {
        "calories": 150, "protein": 7.0, "carbs": 20.0, "fats": 5.5,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "moong_dal": {
        "calories": 105, "protein": 7.2, "carbs": 16.0, "fats": 1.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "rajma": {
        "calories": 127, "protein": 8.7, "carbs": 22.0, "fats": 0.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "chole": {
        "calories": 164, "protein": 8.9, "carbs": 27.0, "fats": 2.6,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "chana_masala": {
        "calories": 164, "protein": 8.9, "carbs": 27.0, "fats": 2.6,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "rajma_chawal": {
        "calories": 135, "protein": 6.5, "carbs": 24.0, "fats": 2.0,
        "portion_g": 350, "serving_desc": "1 plate (rice + rajma)",
    },
    "chole_rice": {
        "calories": 148, "protein": 6.0, "carbs": 26.0, "fats": 2.5,
        "portion_g": 350, "serving_desc": "1 plate (rice + chole)",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MESS — VEGETABLE DISHES
    # ══════════════════════════════════════════════════════════════════════════

    "aloo_gobi": {
        "calories": 90, "protein": 2.5, "carbs": 14.0, "fats": 3.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "gobi": {
        "calories": 75, "protein": 2.5, "carbs": 11.0, "fats": 2.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "phool_gobi": {
        "calories": 75, "protein": 2.5, "carbs": 11.0, "fats": 2.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "fula_gobi": {
        "calories": 75, "protein": 2.5, "carbs": 11.0, "fats": 2.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "gobi_masala": {
        "calories": 90, "protein": 2.8, "carbs": 13.0, "fats": 3.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "gobi_fry": {
        "calories": 110, "protein": 3.0, "carbs": 13.0, "fats": 5.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "besan_aloo_curry": {
        "calories": 105, "protein": 3.5, "carbs": 16.0, "fats": 3.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "soyabean_aloo_curry": {
        "calories": 112, "protein": 6.0, "carbs": 14.0, "fats": 4.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "aloo_chips": {
        "calories": 312, "protein": 3.5, "carbs": 45.0, "fats": 13.0,
        "portion_g": 50, "serving_desc": "1 small packet",
    },
    "jeera_aloo": {
        "calories": 95, "protein": 2.0, "carbs": 15.0, "fats": 3.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "aloo_fry": {
        "calories": 115, "protein": 2.0, "carbs": 17.0, "fats": 4.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "aloo_sabzi": {
        "calories": 100, "protein": 2.0, "carbs": 16.0, "fats": 3.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "aloo_matar": {
        "calories": 105, "protein": 3.5, "carbs": 17.0, "fats": 3.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "bhindi_masala": {
        "calories": 75, "protein": 2.0, "carbs": 10.0, "fats": 3.5,
        "portion_g": 100, "serving_desc": "1 bowl",
    },
    "mix_veg": {
        "calories": 80, "protein": 2.2, "carbs": 12.0, "fats": 2.8,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "mixed_veg": {
        "calories": 80, "protein": 2.2, "carbs": 12.0, "fats": 2.8,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "mix_fry_bhaja": {
        "calories": 130, "protein": 2.5, "carbs": 15.0, "fats": 7.0,
        "portion_g": 100, "serving_desc": "1 small plate",
    },
    "seasonal_bhaja": {
        "calories": 120, "protein": 2.0, "carbs": 14.0, "fats": 6.5,
        "portion_g": 100, "serving_desc": "1 small plate",
    },
    "mixed_boiled_vegetables": {
        "calories": 40, "protein": 2.0, "carbs": 7.0, "fats": 0.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "veg_jaipuri": {
        "calories": 110, "protein": 3.5, "carbs": 14.0, "fats": 4.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "veg_manchurian": {
        "calories": 145, "protein": 3.8, "carbs": 18.0, "fats": 6.5,
        "portion_g": 200, "serving_desc": "1 plate (6 pieces)",
    },
    "gobi_manchurian": {
        "calories": 135, "protein": 3.5, "carbs": 17.0, "fats": 6.0,
        "portion_g": 150, "serving_desc": "1 plate (6 pieces)",
    },
    "malai_kofta": {
        "calories": 200, "protein": 6.0, "carbs": 18.0, "fats": 12.0,
        "portion_g": 200, "serving_desc": "1 bowl (4 koftas)",
    },
    "ambula_rai": {
        "calories": 60, "protein": 1.5, "carbs": 10.0, "fats": 2.0,
        "portion_g": 100, "serving_desc": "1 small bowl",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MESS — PANEER DISHES
    # ══════════════════════════════════════════════════════════════════════════

    "paneer": {
        "calories": 265, "protein": 18.0, "carbs": 2.0, "fats": 21.0,
        "portion_g": 100, "serving_desc": "100g (6–7 cubes)",
    },
    "paneer_butter_masala": {
        "calories": 198, "protein": 10.0, "carbs": 12.0, "fats": 13.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "palak_paneer": {
        "calories": 156, "protein": 9.0, "carbs": 8.0, "fats": 10.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "paneer_bhurji": {
        "calories": 190, "protein": 11.0, "carbs": 5.0, "fats": 14.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "chilli_paneer": {
        "calories": 175, "protein": 9.0, "carbs": 14.0, "fats": 9.5,
        "portion_g": 200, "serving_desc": "1 plate",
    },
    "paneer_hyderabadi": {
        "calories": 210, "protein": 10.0, "carbs": 14.0, "fats": 13.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "paneer_tikka": {
        "calories": 220, "protein": 14.0, "carbs": 8.0, "fats": 15.0,
        "portion_g": 150, "serving_desc": "1 plate (4 pieces)",
    },
    "paneer_manchurian": {
        "calories": 185, "protein": 10.0, "carbs": 14.0, "fats": 10.0,
        "portion_g": 200, "serving_desc": "1 plate",
    },
    "paneer_momos": {
        "calories": 175, "protein": 8.0, "carbs": 22.0, "fats": 6.5,
        "portion_g": 150, "serving_desc": "6 pieces",
    },
    "matar_paneer": {
        "calories": 175, "protein": 9.5, "carbs": 14.0, "fats": 9.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "shahi_paneer": {
        "calories": 220, "protein": 10.0, "carbs": 12.0, "fats": 15.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "kadai_paneer": {
        "calories": 200, "protein": 10.5, "carbs": 11.0, "fats": 13.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MESS — NON-VEG
    # ══════════════════════════════════════════════════════════════════════════

    "fish_masala": {
        "calories": 145, "protein": 18.0, "carbs": 5.0, "fats": 6.0,
        "portion_g": 150, "serving_desc": "1 piece with gravy",
    },
    "fish_besar": {
        "calories": 138, "protein": 17.0, "carbs": 4.0, "fats": 6.0,
        "portion_g": 150, "serving_desc": "1 piece with gravy",
    },
    "chicken_65": {
        "calories": 230, "protein": 18.0, "carbs": 10.0, "fats": 13.0,
        "portion_g": 150, "serving_desc": "1 plate (6 pieces)",
    },
    "chicken_mughlai": {
        "calories": 195, "protein": 20.0, "carbs": 6.0, "fats": 11.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "chicken_curry": {
        "calories": 165, "protein": 18.0, "carbs": 5.0, "fats": 8.5,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "egg_curry": {
        "calories": 145, "protein": 11.0, "carbs": 4.0, "fats": 9.5,
        "portion_g": 150, "serving_desc": "1 bowl (2 eggs)",
    },
    "chicken_tikka": {
        "calories": 195, "protein": 22.0, "carbs": 4.0, "fats": 10.0,
        "portion_g": 150, "serving_desc": "1 plate (4 pieces)",
    },
    "seekh_kebab": {
        "calories": 210, "protein": 18.0, "carbs": 8.0, "fats": 12.0,
        "portion_g": 120, "serving_desc": "2 pieces",
    },
    "anda_bhurji": {
        "calories": 180, "protein": 11.0, "carbs": 3.0, "fats": 13.0,
        "portion_g": 120, "serving_desc": "1 plate (2 eggs)",
    },
    "egg_bhurji": {
        "calories": 180, "protein": 11.0, "carbs": 3.0, "fats": 13.0,
        "portion_g": 120, "serving_desc": "1 plate (2 eggs)",
    },
    "chicken_do_pyaza": {
        "calories": 175, "protein": 19.0, "carbs": 7.0, "fats": 9.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "mutton_curry": {
        "calories": 190, "protein": 20.0, "carbs": 4.0, "fats": 11.0,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "prawn_curry": {
        "calories": 130, "protein": 16.0, "carbs": 5.0, "fats": 6.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MESS — CONDIMENTS & SIDES
    # ══════════════════════════════════════════════════════════════════════════

    "tomato_chutney": {
        "calories": 55, "protein": 1.0, "carbs": 9.0, "fats": 1.5,
        "portion_g": 30, "serving_desc": "2 tablespoons",
    },
    "salad": {
        "calories": 20, "protein": 1.0, "carbs": 4.0, "fats": 0.2,
        "portion_g": 100, "serving_desc": "1 small plate",
    },
    "achaar": {
        "calories": 95, "protein": 1.0, "carbs": 10.0, "fats": 6.0,
        "portion_g": 15, "serving_desc": "1 tablespoon",
    },
    "papad": {
        "calories": 350, "protein": 16.0, "carbs": 59.0, "fats": 5.0,
        "portion_g": 10, "serving_desc": "1 piece",
    },
    "bundi": {
        "calories": 360, "protein": 8.0, "carbs": 50.0, "fats": 14.0,
        "portion_g": 30, "serving_desc": "1 small cup",
    },
    "raita": {
        "calories": 60, "protein": 2.5, "carbs": 6.0, "fats": 2.5,
        "portion_g": 100, "serving_desc": "1 small bowl",
    },
    "chuda_bhaja": {
        "calories": 410, "protein": 7.0, "carbs": 58.0, "fats": 17.0,
        "portion_g": 50, "serving_desc": "1 small bowl",
    },
    "corn_black_chana_sprout": {
        "calories": 105, "protein": 5.5, "carbs": 18.0, "fats": 1.5,
        "portion_g": 100, "serving_desc": "1 small bowl",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MESS — DESSERTS
    # ══════════════════════════════════════════════════════════════════════════

    "gulab_jamun": {
        "calories": 320, "protein": 4.5, "carbs": 55.0, "fats": 10.0,
        "portion_g": 60, "serving_desc": "2 pieces",
    },
    "rasgulla": {
        "calories": 186, "protein": 3.5, "carbs": 43.0, "fats": 0.5,
        "portion_g": 100, "serving_desc": "2 pieces",
    },
    "ice_cream": {
        "calories": 207, "protein": 3.5, "carbs": 24.0, "fats": 11.0,
        "portion_g": 100, "serving_desc": "1 scoop",
    },
    "semiya_kheer": {
        "calories": 150, "protein": 4.0, "carbs": 26.0, "fats": 4.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "rice_kheer": {
        "calories": 145, "protein": 4.0, "carbs": 25.0, "fats": 4.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "amra_sweet": {
        "calories": 130, "protein": 1.0, "carbs": 30.0, "fats": 1.5,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "gajar_halwa": {
        "calories": 210, "protein": 4.0, "carbs": 32.0, "fats": 8.0,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "jalebi": {
        "calories": 370, "protein": 2.0, "carbs": 65.0, "fats": 11.0,
        "portion_g": 80, "serving_desc": "4 pieces",
    },
    "imarti": {
        "calories": 360, "protein": 3.0, "carbs": 63.0, "fats": 10.0,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "rabri": {
        "calories": 215, "protein": 6.0, "carbs": 28.0, "fats": 9.5,
        "portion_g": 150, "serving_desc": "1 bowl",
    },
    "kulfi": {
        "calories": 225, "protein": 5.0, "carbs": 26.0, "fats": 12.0,
        "portion_g": 80, "serving_desc": "1 stick",
    },
    "malai_kulfi": {
        "calories": 230, "protein": 5.5, "carbs": 26.0, "fats": 12.5,
        "portion_g": 80, "serving_desc": "1 stick",
    },
    "paan_kulfi": {
        "calories": 220, "protein": 4.5, "carbs": 27.0, "fats": 11.5,
        "portion_g": 80, "serving_desc": "1 stick",
    },
    "ghewar": {
        "calories": 385, "protein": 5.0, "carbs": 58.0, "fats": 15.0,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "falooda": {
        "calories": 180, "protein": 4.5, "carbs": 33.0, "fats": 4.0,
        "portion_g": 300, "serving_desc": "1 glass",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # STREET FOOD — CHAAT
    # ══════════════════════════════════════════════════════════════════════════

    "pani_puri": {
        "calories": 180, "protein": 3.5, "carbs": 28.0, "fats": 6.0,
        "portion_g": 100, "serving_desc": "6 pieces",
    },
    "bhel_puri": {
        "calories": 155, "protein": 4.0, "carbs": 28.0, "fats": 3.5,
        "portion_g": 150, "serving_desc": "1 plate",
    },
    "sev_puri": {
        "calories": 185, "protein": 4.5, "carbs": 26.0, "fats": 7.5,
        "portion_g": 120, "serving_desc": "5 pieces",
    },
    "dahi_puri": {
        "calories": 195, "protein": 5.5, "carbs": 30.0, "fats": 6.0,
        "portion_g": 150, "serving_desc": "6 pieces",
    },
    "aloo_tikki_chaat": {
        "calories": 190, "protein": 4.5, "carbs": 30.0, "fats": 7.0,
        "portion_g": 150, "serving_desc": "2 tikkis with toppings",
    },
    "papdi_chaat": {
        "calories": 200, "protein": 5.0, "carbs": 30.0, "fats": 7.5,
        "portion_g": 150, "serving_desc": "1 plate",
    },
    "raj_kachori": {
        "calories": 280, "protein": 7.0, "carbs": 38.0, "fats": 11.0,
        "portion_g": 200, "serving_desc": "1 piece",
    },
    "samosa": {
        "calories": 262, "protein": 4.5, "carbs": 30.0, "fats": 14.0,
        "portion_g": 70, "serving_desc": "1 piece",
    },
    "samosa_chaat": {
        "calories": 240, "protein": 6.0, "carbs": 34.0, "fats": 10.0,
        "portion_g": 200, "serving_desc": "1 plate (1 samosa with toppings)",
    },
    "kachori": {
        "calories": 310, "protein": 5.0, "carbs": 38.0, "fats": 15.0,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "pyaz_kachori": {
        "calories": 315, "protein": 5.5, "carbs": 37.0, "fats": 16.0,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "dal_kachori": {
        "calories": 305, "protein": 6.5, "carbs": 36.0, "fats": 14.0,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "jhal_muri": {
        "calories": 120, "protein": 3.0, "carbs": 20.0, "fats": 3.5,
        "portion_g": 100, "serving_desc": "1 small bowl",
    },
    "chaat_corn": {
        "calories": 115, "protein": 2.5, "carbs": 22.0, "fats": 2.5,
        "portion_g": 150, "serving_desc": "1 cup",
    },
    "butter_corn": {
        "calories": 130, "protein": 2.5, "carbs": 21.0, "fats": 4.5,
        "portion_g": 150, "serving_desc": "1 cup",
    },
    "roasted_corn": {
        "calories": 95, "protein": 3.0, "carbs": 19.0, "fats": 1.5,
        "portion_g": 100, "serving_desc": "1 cob",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # STREET FOOD — VEG SNACKS & FRIED
    # ══════════════════════════════════════════════════════════════════════════

    "vada_pav": {
        "calories": 290, "protein": 7.0, "carbs": 42.0, "fats": 10.0,
        "portion_g": 150, "serving_desc": "1 piece",
    },
    "dabeli": {
        "calories": 260, "protein": 6.0, "carbs": 40.0, "fats": 9.0,
        "portion_g": 130, "serving_desc": "1 piece",
    },
    "pav_bhaji": {
        "calories": 220, "protein": 6.0, "carbs": 35.0, "fats": 7.5,
        "portion_g": 300, "serving_desc": "2 pav with 1 bowl bhaji",
    },
    "misal_pav": {
        "calories": 250, "protein": 9.0, "carbs": 38.0, "fats": 7.5,
        "portion_g": 300, "serving_desc": "2 pav with 1 bowl misal",
    },
    "chole_bhature": {
        "calories": 330, "protein": 10.0, "carbs": 48.0, "fats": 12.0,
        "portion_g": 300, "serving_desc": "2 bhature with 1 bowl chole",
    },
    "kulche_chole": {
        "calories": 310, "protein": 10.0, "carbs": 50.0, "fats": 8.0,
        "portion_g": 300, "serving_desc": "2 kulche with 1 bowl chole",
    },
    "bread_pakora": {
        "calories": 265, "protein": 7.0, "carbs": 34.0, "fats": 12.0,
        "portion_g": 100, "serving_desc": "1 piece",
    },
    "aloo_bonda": {
        "calories": 210, "protein": 4.0, "carbs": 28.0, "fats": 9.5,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "mirchi_bajji": {
        "calories": 185, "protein": 3.5, "carbs": 24.0, "fats": 8.5,
        "portion_g": 80, "serving_desc": "1 piece",
    },
    "pakora": {
        "calories": 235, "protein": 5.0, "carbs": 26.0, "fats": 12.0,
        "portion_g": 100, "serving_desc": "4–5 pieces",
    },
    "onion_pakora": {
        "calories": 225, "protein": 4.5, "carbs": 25.0, "fats": 11.5,
        "portion_g": 100, "serving_desc": "4–5 pieces",
    },
    "moong_dal_pakora": {
        "calories": 245, "protein": 8.0, "carbs": 26.0, "fats": 12.0,
        "portion_g": 100, "serving_desc": "4–5 pieces",
    },
    "pasta": {
        "calories": 155, "protein": 5.0, "carbs": 28.0, "fats": 2.5,
        "portion_g": 200, "serving_desc": "1 bowl",
    },
    "white_sauce_pasta": {
        "calories": 190, "protein": 6.5, "carbs": 28.0, "fats": 7.0,
        "portion_g": 250, "serving_desc": "1 bowl",
    },
    "red_sauce_pasta": {
        "calories": 170, "protein": 5.5, "carbs": 30.0, "fats": 4.0,
        "portion_g": 250, "serving_desc": "1 bowl",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # STREET FOOD — ROLLS, WRAPS & BURGERS
    # ══════════════════════════════════════════════════════════════════════════

    "veg_shawarma": {
        "calories": 280, "protein": 7.0, "carbs": 40.0, "fats": 10.0,
        "portion_g": 200, "serving_desc": "1 wrap",
    },
    "paneer_shawarma": {
        "calories": 320, "protein": 12.0, "carbs": 38.0, "fats": 13.0,
        "portion_g": 220, "serving_desc": "1 wrap",
    },
    "chicken_shawarma": {
        "calories": 340, "protein": 18.0, "carbs": 35.0, "fats": 14.0,
        "portion_g": 220, "serving_desc": "1 wrap",
    },
    "egg_shawarma": {
        "calories": 300, "protein": 12.0, "carbs": 37.0, "fats": 12.0,
        "portion_g": 200, "serving_desc": "1 wrap",
    },
    "veg_roll": {
        "calories": 240, "protein": 5.5, "carbs": 38.0, "fats": 7.5,
        "portion_g": 180, "serving_desc": "1 roll",
    },
    "paneer_roll": {
        "calories": 290, "protein": 10.0, "carbs": 37.0, "fats": 11.0,
        "portion_g": 200, "serving_desc": "1 roll",
    },
    "chicken_roll": {
        "calories": 300, "protein": 15.0, "carbs": 35.0, "fats": 11.0,
        "portion_g": 200, "serving_desc": "1 roll",
    },
    "egg_roll": {
        "calories": 265, "protein": 10.0, "carbs": 35.0, "fats": 9.5,
        "portion_g": 180, "serving_desc": "1 roll",
    },
    "kathi_roll": {
        "calories": 285, "protein": 9.0, "carbs": 38.0, "fats": 10.0,
        "portion_g": 200, "serving_desc": "1 roll",
    },
    "frankie": {
        "calories": 270, "protein": 8.0, "carbs": 38.0, "fats": 9.5,
        "portion_g": 180, "serving_desc": "1 roll",
    },
    "egg_kathi_roll": {
        "calories": 275, "protein": 11.0, "carbs": 36.0, "fats": 9.5,
        "portion_g": 180, "serving_desc": "1 roll",
    },
    "veg_burger": {
        "calories": 290, "protein": 7.0, "carbs": 42.0, "fats": 10.0,
        "portion_g": 180, "serving_desc": "1 burger",
    },
    "aloo_tikki_burger": {
        "calories": 295, "protein": 6.5, "carbs": 43.0, "fats": 11.0,
        "portion_g": 180, "serving_desc": "1 burger",
    },
    "chicken_burger": {
        "calories": 360, "protein": 18.0, "carbs": 38.0, "fats": 15.0,
        "portion_g": 200, "serving_desc": "1 burger",
    },
    "veg_sandwich": {
        "calories": 210, "protein": 6.0, "carbs": 32.0, "fats": 6.0,
        "portion_g": 150, "serving_desc": "1 sandwich (2 slices)",
    },
    "grilled_sandwich": {
        "calories": 245, "protein": 8.0, "carbs": 33.0, "fats": 9.0,
        "portion_g": 150, "serving_desc": "1 sandwich (2 slices)",
    },
    "cheese_sandwich": {
        "calories": 265, "protein": 10.0, "carbs": 32.0, "fats": 11.0,
        "portion_g": 150, "serving_desc": "1 sandwich (2 slices)",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # STREET FOOD — CHINESE / INDO-CHINESE
    # ══════════════════════════════════════════════════════════════════════════

    "egg_chowmein": {
        "calories": 185, "protein": 8.0, "carbs": 28.0, "fats": 5.5,
        "portion_g": 250, "serving_desc": "1 plate",
    },
    "chicken_chowmein": {
        "calories": 200, "protein": 12.0, "carbs": 26.0, "fats": 5.5,
        "portion_g": 250, "serving_desc": "1 plate",
    },
    "veg_fried_rice": {
        "calories": 155, "protein": 3.5, "carbs": 28.0, "fats": 3.5,
        "portion_g": 250, "serving_desc": "1 plate",
    },
    "egg_fried_rice": {
        "calories": 175, "protein": 7.0, "carbs": 27.0, "fats": 4.5,
        "portion_g": 250, "serving_desc": "1 plate",
    },
    "chicken_fried_rice": {
        "calories": 190, "protein": 10.0, "carbs": 26.0, "fats": 4.5,
        "portion_g": 250, "serving_desc": "1 plate",
    },
    "chicken_manchurian": {
        "calories": 185, "protein": 14.0, "carbs": 12.0, "fats": 9.0,
        "portion_g": 200, "serving_desc": "1 plate",
    },
    "chilli_chicken": {
        "calories": 210, "protein": 16.0, "carbs": 12.0, "fats": 11.0,
        "portion_g": 200, "serving_desc": "1 plate",
    },
    "hakka_noodles": {
        "calories": 165, "protein": 5.0, "carbs": 28.0, "fats": 4.0,
        "portion_g": 250, "serving_desc": "1 plate",
    },
    "schezwan_noodles": {
        "calories": 175, "protein": 5.0, "carbs": 28.0, "fats": 5.0,
        "portion_g": 250, "serving_desc": "1 plate",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # STREET FOOD — MOMOS
    # ══════════════════════════════════════════════════════════════════════════

    "veg_momos": {
        "calories": 155, "protein": 5.5, "carbs": 22.0, "fats": 5.0,
        "portion_g": 150, "serving_desc": "6 pieces",
    },
    "chicken_momos": {
        "calories": 175, "protein": 10.0, "carbs": 20.0, "fats": 6.0,
        "portion_g": 150, "serving_desc": "6 pieces",
    },
    "tandoori_momos": {
        "calories": 200, "protein": 9.0, "carbs": 22.0, "fats": 8.5,
        "portion_g": 150, "serving_desc": "6 pieces",
    },
    "fried_momos": {
        "calories": 220, "protein": 7.0, "carbs": 24.0, "fats": 11.0,
        "portion_g": 150, "serving_desc": "6 pieces",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # STREET FOOD — DRINKS & SMOOTHIES
    # ══════════════════════════════════════════════════════════════════════════

    "lassi": {
        "calories": 100, "protein": 3.5, "carbs": 12.0, "fats": 4.0,
        "portion_g": 250, "serving_desc": "1 glass",
    },
    "sweet_lassi": {
        "calories": 115, "protein": 3.5, "carbs": 16.0, "fats": 4.0,
        "portion_g": 250, "serving_desc": "1 glass",
    },
    "salted_lassi": {
        "calories": 80, "protein": 3.5, "carbs": 7.0, "fats": 4.0,
        "portion_g": 250, "serving_desc": "1 glass",
    },
    "mango_lassi": {
        "calories": 130, "protein": 3.0, "carbs": 22.0, "fats": 3.5,
        "portion_g": 250, "serving_desc": "1 glass",
    },
    "buttermilk": {
        "calories": 40, "protein": 3.0, "carbs": 5.0, "fats": 1.0,
        "portion_g": 250, "serving_desc": "1 glass",
    },
    "sugarcane_juice": {
        "calories": 65, "protein": 0.2, "carbs": 16.0, "fats": 0.1,
        "portion_g": 250, "serving_desc": "1 glass",
    },
    "lemon_soda": {
        "calories": 35, "protein": 0.1, "carbs": 9.0, "fats": 0.0,
        "portion_g": 300, "serving_desc": "1 glass",
    },
    "jaljeera": {
        "calories": 30, "protein": 0.3, "carbs": 7.0, "fats": 0.2,
        "portion_g": 200, "serving_desc": "1 glass",
    },
    "cold_coffee": {
        "calories": 130, "protein": 4.0, "carbs": 18.0, "fats": 5.0,
        "portion_g": 300, "serving_desc": "1 glass",
    },
    "fruit_smoothie": {
        "calories": 110, "protein": 2.0, "carbs": 24.0, "fats": 1.0,
        "portion_g": 300, "serving_desc": "1 glass",
    },
    "banana_smoothie": {
        "calories": 120, "protein": 3.0, "carbs": 25.0, "fats": 2.0,
        "portion_g": 300, "serving_desc": "1 glass",
    },
    "mango_smoothie": {
        "calories": 125, "protein": 2.0, "carbs": 27.0, "fats": 1.5,
        "portion_g": 300, "serving_desc": "1 glass",
    },
    "strawberry_smoothie": {
        "calories": 105, "protein": 2.5, "carbs": 22.0, "fats": 1.5,
        "portion_g": 300, "serving_desc": "1 glass",
    },
    "chocolate_smoothie": {
        "calories": 180, "protein": 5.0, "carbs": 28.0, "fats": 6.0,
        "portion_g": 300, "serving_desc": "1 glass",
    },
    "protein_smoothie": {
        "calories": 200, "protein": 20.0, "carbs": 20.0, "fats": 4.0,
        "portion_g": 350, "serving_desc": "1 large glass",
    },
}


# ── Convenience helpers ───────────────────────────────────────────────────────

def get_all_keys() -> list[str]:
    return list(NUTRITION_DB.keys())


def get_entry(key: str) -> dict | None:
    return NUTRITION_DB.get(key)


def build_result(key: str) -> dict:
    """Scale a per-100g entry to its standard portion and return a result dict."""
    entry     = NUTRITION_DB[key]
    portion_g = entry["portion_g"]
    scale     = portion_g / 100.0
    return {
        "dish"        : key,
        "calories"    : round(entry["calories"] * scale, 1),
        "protein"     : round(entry["protein"]  * scale, 1),
        "carbs"       : round(entry["carbs"]    * scale, 1),
        "fats"        : round(entry["fats"]     * scale, 1),
        "portion_g"   : portion_g,
        "serving_desc": entry.get("serving_desc", "1 serving"),
        "source"      : "lookup_table",
    }