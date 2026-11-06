"""語板に無い語（固有名詞・複合語）をクラスに当てる表と引き方（介入の研究からの注文 3、2026-11-06）

出す物: proper_class_table.json
  pos: UniDic の細かい品詞（GiNZA の tag_）→ 帯 D の割合（語板 v2 の D と同じ形: 上位のクラスに等分）
  ne:  GiNZA の固有表現の種類（ent_type_）→ 帯 D の割合
引き方（lookup）:
  1. 語板にある語 → その語板。
  2. UniDic の品詞が人名・地名なら品詞の表（後ろの部分や固有表現より確か: 浦島 → 島、桃太郎 → 食べ物 の誤りを防ぐ）。
  3. 複合語は後ろの部分（主要部）で引く: 見出しの後ろから、語板にある最も長い部分（2 文字以上、無ければ 1 文字の名詞）。
     白雪姫 → 姫、玉手箱 → 箱、逃げ帰る → 帰る。動詞は「〜る/〜う」で終わる部分だけを動詞の語板から探す。
  4. 固有表現の種類（ne）→ 表。5. UniDic の品詞（pos）→ 表。6. どれも無ければ None。
クラスの名前は語板 v2（動詞のクラスを戻した版、D = 名詞 80 + 動詞 20）の名前に合わせてある。
実行: python lexicon/proper_class_table.py --out proper_class_table.json [--boards content_boards.json]（--boards があれば表の名前が D の名札にあるか確かめる）
"""
from __future__ import annotations

import argparse
import json

PERSON = ["個人(individual)", "生物(being)", "生き物(living_thing)"]
PLACE = ["地域(region)", "地理的地域(geographical_region)"]
COUNTRY = ["地区(country)", "地域(region)", "地理的地域(geographical_region)"]
POINT = ["場所(point)"]
ORG = ["オーガナイゼーション(organization)", "社会的単位(social_unit)"]
ANIMAL = ["四つ脚(animate_being)", "生物(being)", "生き物(living_thing)"]
MAMMAL = ["哺乳(mammalian)", "脊椎動物(vertebrate)", "脊索動物(chordate)", "四つ脚(animate_being)", "生物(being)", "生き物(living_thing)"]
VERTEBRATE = ["脊椎動物(vertebrate)", "脊索動物(chordate)", "四つ脚(animate_being)", "生物(being)", "生き物(living_thing)"]
PLANT = ["生物(being)", "生き物(living_thing)"]
FOOD = ["滋養分(food)", "物質(substance)"]
BUILDING = ["営造物(structure)", "アーチファクト(artefact)"]
VEHICLE = ["乗物(vehicle)", "輸送機関(conveyance)", "道具(instrumentality)", "アーチファクト(artefact)"]
GOODS = ["消費財(consumer_goods)", "産品(product)", "アーチファクト(artefact)"]
TOOL = ["道具(instrumentality)", "アーチファクト(artefact)"]
PRINT = ["刊行物(publication)", "著作(writing)", "創造物(creation)"]
WORK = ["ワーク(work)", "創造物(creation)", "産品(product)"]
EVENT = ["催し物(social_event)", "出来事(occurrence)"]
TIME = ["ピリオド(period)"]
NATURE = ["自然物(natural_object)"]
PHEN = ["自然現象(natural_phenomenon)", "物理現象(physical_phenomenon)"]
MATERIAL = ["マティエール(material)", "物質(substance)"]
BODY = ["身体部位(body_part)"]
IDEA = ["考え(idea)", "概念(conception)", "認知的内容(mental_object)"]
LANG = ["自然言語(tongue)", "言語単位(linguistic_unit)"]

POS = {
    "名詞-固有名詞-人名-一般": PERSON, "名詞-固有名詞-人名-姓": PERSON, "名詞-固有名詞-人名-名": PERSON,
    "名詞-固有名詞-地名-一般": PLACE, "名詞-固有名詞-地名-国": COUNTRY,
    "名詞-固有名詞-一般": None,   # 種類が分からない（固有表現の種類があればそちら）
}
NE = {
    # 人・生き物
    "Person": PERSON, "God": PERSON, "Character": PERSON, "N_Person": PERSON, "Position_Vocation": PERSON, "Ethnic_Group_Other": PERSON,
    "Nationality": PERSON, "Family": PERSON,
    "Mammal": MAMMAL, "Bird": VERTEBRATE, "Fish": VERTEBRATE, "Reptile": VERTEBRATE, "Amphibia": VERTEBRATE,
    "Insect": ANIMAL, "Mollusc_Arthropod": ANIMAL, "Living_Thing_Other": ANIMAL, "N_Animal": ANIMAL,
    "Flora": PLANT, "Fungus": PLANT, "N_Flora": PLANT,
    "Animal_Part": BODY, "Flora_Part": NATURE, "Living_Thing_Part_Other": BODY,
    # 場所
    "City": PLACE, "Province": PLACE, "County": PLACE, "Domestic_Region": PLACE, "Continental_Region": PLACE, "Region_Other": PLACE,
    "GPE_Other": PLACE, "GOE_Other": BUILDING, "Location_Other": PLACE, "N_Location_Other": PLACE,
    "Country": COUNTRY, "N_Country": COUNTRY,
    "Mountain": NATURE + PLACE, "Island": NATURE + PLACE, "River": NATURE, "Lake": NATURE, "Sea": NATURE + PLACE, "Bay": NATURE,
    "Geological_Region_Other": NATURE + PLACE, "Astral_Body_Other": NATURE, "Planet": NATURE, "Constellation": NATURE,
    "Natural_Object_Other": NATURE, "N_Natural_Object_Other": NATURE, "Mineral": MATERIAL, "Element": MATERIAL, "Compound": MATERIAL,
    "Material": MATERIAL,
    "Facility_Other": BUILDING, "Facility_Part": BUILDING, "N_Facility": BUILDING, "Worship_Place": BUILDING, "Museum": BUILDING,
    "School": BUILDING + ORG, "Station": BUILDING, "Airport": BUILDING, "Port": BUILDING, "Car_Stop": BUILDING, "Park": BUILDING,
    "Sports_Facility": BUILDING, "Theater": BUILDING, "Zoo": BUILDING, "Amusement_Park": BUILDING, "Spa": BUILDING, "Bridge": BUILDING,
    "Tunnel": BUILDING, "Road": BUILDING, "Railroad": BUILDING, "Canal": BUILDING, "Water_Route": BUILDING, "Line_Other": BUILDING,
    "Tumulus": BUILDING, "Archaeological_Place_Other": BUILDING, "Research_Institute": BUILDING + ORG,
    "Postal_Address": POINT, "Point": POINT, "Latitude_Longtitude": POINT,
    # 組織
    "Company": ORG, "Company_Group": ORG, "Corporation_Other": ORG, "Organization_Other": ORG, "N_Organization": ORG,
    "International_Organization": ORG, "Government": ORG, "Public_Institution": ORG, "Political_Organization_Other": ORG,
    "Political_Party": ORG, "Cabinet": ORG, "Military": ORG, "Sports_Organization_Other": ORG, "Pro_Sports_Organization": ORG,
    "Sports_League": ORG, "Show_Organization": ORG,
    # 物
    "Food_Other": FOOD, "Dish": FOOD, "Drug": FOOD + GOODS,
    "Product_Other": GOODS, "N_Product": GOODS, "Clothing": GOODS, "Money_Form": GOODS, "Weapon": TOOL, "Art_Other": WORK,
    "Car": VEHICLE, "Train": VEHICLE, "Aircraft": VEHICLE, "Spaceship": VEHICLE, "Ship": VEHICLE, "Vehicle_Other": VEHICLE,
    "Book": PRINT, "Newspaper": PRINT, "Magazine": PRINT, "Printing_Other": PRINT, "Movie": WORK, "Music": WORK, "Show": WORK,
    "Broadcast_Program": WORK, "Game": WORK,
    # 出来事・時・考え
    "Event_Other": EVENT, "N_Event": EVENT, "Occasion_Other": EVENT, "Religious_Festival": EVENT, "Conference": EVENT, "Sport": EVENT,
    "War": EVENT, "Incident_Other": EVENT, "Offense": EVENT, "Earthquake": PHEN, "Natural_Disaster": PHEN,
    "Natural_Phenomenon_Other": PHEN, "Disease_Other": ["容態(condition)"], "Animal_Disease": ["容態(condition)"],
    "Era": TIME, "Date": TIME, "Day_Of_Week": TIME, "Time": TIME, "Period_Day": TIME, "Period_Week": TIME, "Period_Month": TIME,
    "Period_Year": TIME, "Period_Time": TIME, "Periodx_Other": TIME, "Timex_Other": TIME, "Time_Top_Other": TIME,
    "Religion": IDEA, "Doctrine_Method_Other": IDEA, "Theory": IDEA, "Law": IDEA, "Rule_Other": IDEA, "Treaty": IDEA, "Plan": IDEA,
    "Culture": IDEA, "Style": IDEA, "Movement": IDEA, "Academic": IDEA, "Language_Other": LANG, "National_Language": LANG,
    "Award": GOODS, "Title_Other": IDEA, "Name_Other": IDEA, "Class": IDEA, "Rank": IDEA,
    # 数・量（クラスに当てない）
    "Age": None, "Calorie": None, "Countx_Other": None, "Frequency": None, "ID_Number": None, "Intensity": None, "Measurement_Other": None,
    "Money": None, "Multiplication": None, "Numex_Other": None, "Ordinal_Number": None, "Percent": None, "Physical_Extent": None,
    "Phone_Number": None, "Email": None, "URL": None, "School_Age": None, "Space": None, "Speed": None, "Temperature": None,
    "Unit_Other": None, "Volume": None, "Weight": None, "Color_Other": None, "Nature_Color": None,
}


def dist(classes):
    """クラスの並び → 帯 D の割合（等分。語板 v2 の D と同じ）。"""
    if not classes:
        return None
    u = list(dict.fromkeys(classes))
    return {f"クラス:{c}": round(1 / len(u), 4) for c in u}


def lookup(lemma, xpos, ne, boards_by_name, pos_hint=None):
    """語のクラスの割合（帯 D）と、どの規則で引いたか。boards_by_name: {見出し: 語板}。"""
    b = boards_by_name.get(lemma)
    if b is not None and "D" in b["attrs"]:
        return b["attrs"]["D"], "語板"
    if xpos and ("人名" in xpos or "地名" in xpos) and POS.get(xpos):
        return dist(POS[xpos]), f"品詞:{xpos}"            # 人名・地名は品詞を先に（浦島 → 島、桃太郎 → 食べ物 を防ぐ）
    is_verb = pos_hint == "動詞" or (xpos or "").startswith("動詞")
    for k in range(1, len(lemma)):
        tail = lemma[k:]
        if len(tail) < 2 and (is_verb or k < len(lemma) - 1):
            continue
        tb = boards_by_name.get(tail)
        if tb is not None and "D" in tb["attrs"] and (tb["pos"] == "動詞") == is_verb:
            return tb["attrs"]["D"], f"後ろの部分:{tail}"
    if ne and NE.get(ne):
        return dist(NE[ne]), f"固有表現:{ne}"
    if xpos and POS.get(xpos):
        return dist(POS[xpos]), f"品詞:{xpos}"
    return None, "なし"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="proper_class_table.json")
    ap.add_argument("--boards", default=None)
    a = ap.parse_args()
    table = dict(pos={k: dist(v) for k, v in POS.items()}, ne={k: dist(v) for k, v in NE.items()},
                 規則="1 語板 → 2 品詞が人名・地名なら品詞の表 → 3 後ろの部分（2 文字以上、名詞は最後の 1 文字も可。動詞は動詞の語板だけ）→ 4 固有表現の種類 → 5 UniDic の品詞")
    if a.boards:
        boards = json.load(open(a.boards, encoding="utf-8"))
        have = {x for b in boards for x in b["attrs"].get("D", {})}
        miss = sorted({c for d in list(table["pos"].values()) + list(table["ne"].values()) if d for c in d if c not in have})
        table["D に無い名前"] = miss
        print("D に無い名前", miss)
    json.dump(table, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
