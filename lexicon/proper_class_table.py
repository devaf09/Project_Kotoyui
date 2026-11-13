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
表は種類 → 語義で持ち、語板の版ごとに slots.json のクラスへ当てて書き出す（版でクラスが変わっても合う。語板の D と同じ当て方）。
実行: python lexicon/proper_class_table.py --slots <語板の置き場>/slots.json --wordnet wnjpn.db --out <語板の置き場>/proper_class_table.json
"""
from __future__ import annotations

import argparse
import json

# 種類 → 日本語 WordNet の語義（版によって選ばれるクラスが違うので、クラスの名前ではなく語義で持ち、
# 語板の slots.json のクラスに合わせて「語義の上位にある選ばれたクラス」に当てる。語板の D と同じ当て方）
PERSON = ["00007846-n"]                 # individual（人）
PLACE = ["00027167-n"]                  # location
COUNTRY = ["08544813-n"]                # state（国）
POINT = ["00027167-n"]
ORG = ["08008335-n"]                    # organization
ANIMAL = ["00015388-n"]                 # animate_being
MAMMAL = ["01861778-n"]
VERTEBRATE = ["01503061-n"]             # bird（鳥・魚・爬虫類は下で個別に）
FISH, REPTILE, AMPHIB, INSECT = ["02512053-n"], ["01661091-n"], ["01627424-n"], ["02159955-n"]
PLANT = ["00017222-n"]                  # flora
FOOD = ["07555863-n"]
DRINK = ["07881800-n"]
BUILDING = ["02913152-n"]               # edifice
STRUCT = ["04341686-n"]                 # structure（道・橋・施設）
VEHICLE = ["04524313-n"]
GOODS = ["03076708-n"]                  # commodity
TOOL = ["03575240-n"]                   # instrumentality
WEAPON = ["04565375-n"]
CLOTHING = ["03051540-n"]
PRINT = ["06589574-n"]                  # publication
WORK = ["00933420-n"]                   # artistic_production
EVENT = ["07288639-n"]                  # social_event
TIME = ["15113229-n"]                   # period
NATURE = ["00019128-n"]                 # natural_object
WATER = ["09225146-n"]
SKY = ["09239740-n"]
PHEN = ["11408559-n"]
MATERIAL = ["14580897-n"]
BODY = ["05220461-n"]
IDEA = ["05833840-n"]
LANG = ["06282651-n"]
CONDITION = ["13920835-n"]
MONEY = ["13384557-n"]

POS = {
    "名詞-固有名詞-人名-一般": PERSON, "名詞-固有名詞-人名-姓": PERSON, "名詞-固有名詞-人名-名": PERSON,
    "名詞-固有名詞-地名-一般": PLACE, "名詞-固有名詞-地名-国": COUNTRY,
    "名詞-固有名詞-一般": None,   # 種類が分からない（固有表現の種類があればそちら）
}
NE = {
    # 人・生き物
    "Person": PERSON, "God": PERSON, "Character": PERSON, "N_Person": PERSON, "Position_Vocation": PERSON, "Ethnic_Group_Other": PERSON,
    "Nationality": PERSON, "Family": PERSON,
    "Mammal": MAMMAL, "Bird": VERTEBRATE, "Fish": FISH, "Reptile": REPTILE, "Amphibia": AMPHIB,
    "Insect": INSECT, "Mollusc_Arthropod": ANIMAL, "Living_Thing_Other": ANIMAL, "N_Animal": ANIMAL,
    "Flora": PLANT, "Fungus": PLANT, "N_Flora": PLANT,
    "Animal_Part": BODY, "Flora_Part": NATURE, "Living_Thing_Part_Other": BODY,
    # 場所
    "City": PLACE, "Province": PLACE, "County": PLACE, "Domestic_Region": PLACE, "Continental_Region": PLACE, "Region_Other": PLACE,
    "GPE_Other": PLACE, "GOE_Other": BUILDING, "Location_Other": PLACE, "N_Location_Other": PLACE,
    "Country": COUNTRY, "N_Country": COUNTRY,
    "Mountain": NATURE, "Island": NATURE, "River": WATER, "Lake": WATER, "Sea": WATER, "Bay": WATER,
    "Geological_Region_Other": NATURE, "Astral_Body_Other": SKY, "Planet": SKY, "Constellation": SKY,
    "Natural_Object_Other": NATURE, "N_Natural_Object_Other": NATURE, "Mineral": MATERIAL, "Element": MATERIAL, "Compound": MATERIAL,
    "Material": MATERIAL,
    "Facility_Other": BUILDING, "Facility_Part": BUILDING, "N_Facility": BUILDING, "Worship_Place": BUILDING, "Museum": BUILDING,
    "School": BUILDING, "Station": BUILDING, "Airport": BUILDING, "Port": BUILDING, "Car_Stop": BUILDING, "Park": BUILDING,
    "Sports_Facility": BUILDING, "Theater": BUILDING, "Zoo": BUILDING, "Amusement_Park": BUILDING, "Spa": BUILDING, "Bridge": STRUCT,
    "Tunnel": STRUCT, "Road": STRUCT, "Railroad": STRUCT, "Canal": STRUCT, "Water_Route": STRUCT, "Line_Other": STRUCT,
    "Tumulus": BUILDING, "Archaeological_Place_Other": BUILDING, "Research_Institute": ORG,
    "Postal_Address": POINT, "Point": POINT, "Latitude_Longtitude": POINT,
    # 組織
    "Company": ORG, "Company_Group": ORG, "Corporation_Other": ORG, "Organization_Other": ORG, "N_Organization": ORG,
    "International_Organization": ORG, "Government": ORG, "Public_Institution": ORG, "Political_Organization_Other": ORG,
    "Political_Party": ORG, "Cabinet": ORG, "Military": ORG, "Sports_Organization_Other": ORG, "Pro_Sports_Organization": ORG,
    "Sports_League": ORG, "Show_Organization": ORG,
    # 物
    "Food_Other": FOOD, "Dish": FOOD, "Drug": GOODS,
    "Product_Other": GOODS, "N_Product": GOODS, "Clothing": CLOTHING, "Money_Form": MONEY, "Weapon": WEAPON, "Art_Other": WORK,
    "Car": VEHICLE, "Train": VEHICLE, "Aircraft": VEHICLE, "Spaceship": VEHICLE, "Ship": VEHICLE, "Vehicle_Other": VEHICLE,
    "Book": PRINT, "Newspaper": PRINT, "Magazine": PRINT, "Printing_Other": PRINT, "Movie": WORK, "Music": WORK, "Show": WORK,
    "Broadcast_Program": WORK, "Game": WORK,
    # 出来事・時・考え
    "Event_Other": EVENT, "N_Event": EVENT, "Occasion_Other": EVENT, "Religious_Festival": EVENT, "Conference": EVENT, "Sport": EVENT,
    "War": EVENT, "Incident_Other": EVENT, "Offense": EVENT, "Earthquake": PHEN, "Natural_Disaster": PHEN,
    "Natural_Phenomenon_Other": PHEN, "Disease_Other": CONDITION, "Animal_Disease": CONDITION,
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


def make_dist(wn, classes):
    """語義の並び → 帯 D の割合を作る関数。classes: {クラスの名前: 語義}（語板の slots.json の classes）。
    語義の上位にある選ばれたクラスに等分する（語板の D と同じ）。"""
    cl = {v: k for k, v in classes.items()}

    def dist(syns):
        if not syns:
            return None
        c = {}
        for s in syns:
            anc = [a for a in wn.ancestors(s) if a in cl]
            for a in anc:
                c[a] = c.get(a, 0) + 1 / len(syns) / len(anc)
        return {f"クラス:{cl[a]}": round(v, 4) for a, v in sorted(c.items(), key=lambda kv: -kv[1])} or None
    return dist


def lookup(lemma, xpos, ne, boards_by_name, table, pos_hint=None):
    """語のクラスの割合（帯 D）と、どの規則で引いたか。boards_by_name: {見出し: 語板}、table: 書き出した表（pos・ne）。"""
    b = boards_by_name.get(lemma)
    if b is not None and "D" in b["attrs"]:
        return b["attrs"]["D"], "語板"
    if xpos and ("人名" in xpos or "地名" in xpos) and table["pos"].get(xpos):
        return table["pos"][xpos], f"品詞:{xpos}"            # 人名・地名は品詞を先に（浦島 → 島、桃太郎 → 食べ物 を防ぐ）
    is_verb = pos_hint == "動詞" or (xpos or "").startswith("動詞")
    for k in range(1, len(lemma)):
        tail = lemma[k:]
        if len(tail) < 2 and (is_verb or k < len(lemma) - 1):
            continue
        tb = boards_by_name.get(tail)
        if tb is not None and "D" in tb["attrs"] and (tb["pos"] == "動詞") == is_verb:
            return tb["attrs"]["D"], f"後ろの部分:{tail}"
    if ne and table["ne"].get(ne):
        return table["ne"][ne], f"固有表現:{ne}"
    if xpos and table["pos"].get(xpos):
        return table["pos"][xpos], f"品詞:{xpos}"
    return None, "なし"


def main():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from content_boards import WordNet
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", required=True, help="語板の slots.json（classes = クラスの名前 → 語義）")
    ap.add_argument("--wordnet", required=True)
    ap.add_argument("--out", default="proper_class_table.json")
    a = ap.parse_args()
    classes = json.load(open(a.slots, encoding="utf-8"))["classes"]
    dist = make_dist(WordNet(a.wordnet), classes)
    table = dict(pos={k: dist(v) for k, v in POS.items()}, ne={k: dist(v) for k, v in NE.items()},
                 語板=os.path.abspath(a.slots),
                 規則="1 語板 → 2 品詞が人名・地名なら品詞の表 → 3 後ろの部分（2 文字以上、名詞は最後の 1 文字も可。動詞は動詞の語板だけ）→ 4 固有表現の種類 → 5 UniDic の品詞")
    miss = sorted(k for k, v in list(POS.items()) + list(NE.items()) if v and not dist(v))
    table["クラスに当たらない種類"] = miss
    print("クラスに当たらない種類", miss)
    json.dump(table, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
