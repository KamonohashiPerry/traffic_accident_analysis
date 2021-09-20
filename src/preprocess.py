import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import re
from decimal import Decimal, ROUND_HALF_UP


def multiple_replace(text, adict):
    '''
    複数の文字列を指定して置換する関数
    '''
    rx = re.compile('|'.join(adict))
    def dedictkey(text):
        for key in adict.keys():
            if re.search(key, text):
                return key

    def one_xlat(match):
        return adict[dedictkey(match.group(0))]

    return rx.sub(one_xlat, text)

def dms2deg(dms):    
    # 度分秒から度への変換
    if len(str(dms)) == 10:        
        h = int(str(dms)[0:3])
        m = int(str(dms)[3:5])
        s = int(str(dms)[5:7])
    elif len(str(dms)) == 9:
        h = int(str(dms)[0:2])
        m = int(str(dms)[2:4])
        s = int(str(dms)[4:6])
    deg = Decimal(str(h + (m / 60) + (s / 3600))).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    return str(deg)

def latlon2mesh(lat, lon):
    #1次メッシュ上2けた
    quotient_lat, remainder_lat = divmod(lat * 60, 40)
    first2digits = str(quotient_lat)[0:2]

    #1次メッシュ下2けた
    last2digits = str(lon - 100)[0:2]
    remainder_lon = lon - int(last2digits) - 100

    #1次メッシュ
    first_mesh = first2digits + last2digits

    #2次メッシュ上1けた
    first1digits, remainder_lat = divmod(remainder_lat, 5)

    #2次メッシュ下1けた
    last1digits, remainder_lon = divmod(remainder_lon * 60, 7.5)

    #2次メッシュ
    second_mesh = first_mesh + str(first1digits)[0:1] + str(last1digits)[0:1]

    #3次メッシュ上1けた
    first1digits, remainder_lat = divmod(remainder_lat * 60, 30)

    #3次メッシュ下1けた
    last1digits, remainder_lon = divmod(remainder_lon * 60, 45)

    #3次メッシュ
    third_mesh = second_mesh + str(first1digits)[0:1] + str(last1digits)[0:1]
    
    return third_mesh


# データの読み込み
traffic_accident_data = pd.read_csv("https://www.npa.go.jp/publications/statistics/koutsuu/opendata/2020/honhyo_2020.csv",
                                                        encoding="shift-jis")
traffic_accident_data_19 = pd.read_csv("https://www.npa.go.jp/publications/statistics/koutsuu/opendata/2019/honhyo_2019.csv",
                                                        encoding="shift-jis")

traffic_accident_data = traffic_accident_data.append(traffic_accident_data_19).reset_index(drop=True)


traffic_accident_data["accident_date"] = pd.to_datetime(traffic_accident_data[['発生日時　　年', '発生日時　　月',
                                                                           '発生日時　　日', '発生日時　　時', '発生日時　　分']].\
                                                                            apply(lambda x: '{}-{}-{} {}:{}'.format(x[0], x[1], x[2], x[3], x[4]), axis=1))

pref_code_df = pd.read_csv("../data/pref_code_conv.csv")
pref_code_df = pref_code_df.rename(columns={'pref_cd':'都道府県コード'})


traffic_accident_data = pd.merge(traffic_accident_data, pref_code_df,
                                                     on="都道府県コード", how="left")

traffic_accident_data["tiiki-code"] = traffic_accident_data[['genuine_pref_cd', '市区町村コード']]\
                                                            .apply(lambda x: '{}{}'.format(x[0], x[1]), axis=1)

city_code_df = pd.read_csv("https://www.soumu.go.jp/main_content/000608358.csv", encoding="shift-jis")
city_code_df = city_code_df.rename(columns={"sityouson-code":'市区町村コード'})
city_code_df["tiiki-code"] = city_code_df["tiiki-code"].astype(str)

traffic_accident_data = pd.merge(traffic_accident_data,city_code_df.drop(columns=["ken-code", "市区町村コード", "yomigana"]),
                                                                            on="tiiki-code", how="left")


traffic_accident_data["accident_type"] = np.where(traffic_accident_data["事故内容"] == 1, "死亡", 
                                                    np.where(traffic_accident_data["事故内容"] == 2, "負傷", None))


traffic_accident_data["road_cd_f4"] = list(map(lambda text:text[0:4], traffic_accident_data["路線コード"].astype(str)))
traffic_accident_data["road_cd_f4"] = traffic_accident_data.road_cd_f4.astype(int)

traffic_accident_data["road_cd_l1"] = list(map(lambda text:text[-1], traffic_accident_data["路線コード"].astype(str)))
traffic_accident_data["road_cd_l1"] = traffic_accident_data.road_cd_l1.astype(int)

traffic_accident_data["road_type"] = np.where(traffic_accident_data.road_cd_f4 <= 999,
                                                                        "一般国道（国道番号）",
                                                                          traffic_accident_data.road_cd_f4)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 1000) & (traffic_accident_data.road_cd_f4 <= 1500),
                                                                        "主要地方道－都道府県道",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 1500) & (traffic_accident_data.road_cd_f4 <= 1999),
                                                                        "主要地方道－市道",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 2000) & (traffic_accident_data.road_cd_f4 <= 2999),
                                                                        "一般都道府県道",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 3000) & (traffic_accident_data.road_cd_f4 <= 3999),
                                                                        "一般市町村道",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 4000) & (traffic_accident_data.road_cd_f4 <= 4999),
                                                                        "高速自動車国道",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 5000) & (traffic_accident_data.road_cd_f4 <= 5499),
                                                                        "自動車専用道－指定",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 5500) & (traffic_accident_data.road_cd_f4 <= 5999),
                                                                        "自動車専用道－その他",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 6000) & (traffic_accident_data.road_cd_f4 <= 6999),
                                                                        "道路運送法上の道路",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 7000) & (traffic_accident_data.road_cd_f4 <= 7999),
                                                                        "農（免）道",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 8000) & (traffic_accident_data.road_cd_f4 <= 8499),
                                                                        "林道",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 8500) & (traffic_accident_data.road_cd_f4 <= 8999),
                                                                        "港湾道",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( (traffic_accident_data.road_cd_f4 >= 9000) & (traffic_accident_data.road_cd_f4 <= 9499),
                                                                        "私道",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( traffic_accident_data.road_cd_f4 == 9500,
                                                                        "その他",
                                                                          traffic_accident_data.road_type)
traffic_accident_data["road_type"] = np.where( traffic_accident_data.road_cd_f4 == 9900,
                                                                        "一般の交通の用に供するその他の道路",
                                                                          traffic_accident_data.road_type)

traffic_accident_data["road_bypass"] = np.where(traffic_accident_data.road_cd_l1 > 0, "バイパス区間",
                                                                           np.where(traffic_accident_data.road_cd_l1 == 0, "現道区間又は包括路線", None))


replace_list = {'1':'上',
                      '2':'下',
                      '0':'対象外'
                        }

traffic_accident_data['road_updown_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['上下線'].astype(str)))


# 昼夜
replace_list = {'11':'昼－明',
                      '12':' 昼－昼',
                      '13':' 昼－暮',
                      '21':'夜－暮',
                      '22':'夜－夜',
                      '23':'夜－明'
                        }

traffic_accident_data['day_night_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['昼夜'].astype(str)))


# 天候
replace_list = {'1':'晴',
                      '2':'曇',
                      '3':'雨',
                      '4':'霧',
                      '5':'雪'
                        }

traffic_accident_data['weather_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['天候'].astype(str)))


# 地形
replace_list = {'1':'市街地－人口集中',
                      '2':'市街地－その他',
                      '3':'非市街地',
                        }

traffic_accident_data['terrain_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['地形'].astype(str)))

# 路面状態
replace_list = {'1':'舗装－乾燥',
                      '2':'舗装－湿潤',
                      '3':'舗装－凍結',
                      '4':'舗装－積雪',
                      '5':'非舗装',
                        }

traffic_accident_data['road_condition_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['路面状態'].astype(str)))


# 道路形状
replace_list = {'31':'交差点－環状交差点',
                      '^1$':'交差点－その他',
                      '37':'交差点付近－環状交差点付近',
                      '^7$':'交差点付近－その他',
                      '11':'単路－トンネル',
                      '12':'単路－橋',
                      '13':'単路－カーブ・屈折',
                      '14':'単路－その他',
                      '21':'踏切－第一種',
                      '22':'踏切－第三種',
                      '23':'踏切－第四種',
                      '^0$':'一般交通の場所'
                        }

traffic_accident_data['road_shape_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['道路形状'].astype(str)))


# 環状交差点の直径
replace_list = {'^1$':'小（27ｍ未満）',
                      '^2$':'中（27ｍ以上）',
                      '^3$':'大（43ｍ以上）',
                      '^0$':'環状交差点以外',
                        }

traffic_accident_data['roundabout_diameter_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['環状交差点の直径'].astype(str)))

# 信号機
replace_list = {'^1$':'点灯－３灯式',
                      '^8$':'点灯－歩車分式',
                      '^2$':'点灯－押ボタン式',
                      '^3$':'点滅－３灯式',
                      '^4$':'点滅－１灯式',
                      '^5$':'消灯',
                      '^6$':'故障',
                      '^7$':'施設なし'
                        }

traffic_accident_data['traffic_lights_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['信号機'].astype(str)))


# 一時停止規制　標識
replace_list = {'^1$':'標準－反射式',
                      '^2$':'標準－自発光式',
                      '^3$':'標準－内部照明式',
                      '^4$':'拡大－反射式',
                      '^5$':'拡大－自発光式',
                      '^6$':'拡大－内部照明式',
                      '^7$':'縮小',
                      '^8$':'その他',
                      '^9$':'規制なし',
                      '^0$':'対象外当事者',
                        }

traffic_accident_data['pause_sign_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['一時停止規制\u3000標識（当事者A）'].astype(str)))
traffic_accident_data['pause_sign_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['一時停止規制\u3000標識（当事者B）'].astype(str)))

#  一時停止規制　表示
replace_list = {'^21$':'表示あり',
                      '^22$':'表示なし',
                      '^23$':'その他',
                        }

traffic_accident_data['pause_display_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['一時停止規制\u3000表示（当事者A）'].astype(str)))
traffic_accident_data['pause_display_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['一時停止規制\u3000表示（当事者B）'].astype(str)))

# 車道幅員
replace_list = {'^1$':'単路－3.5m未満',
                      '^2$':'単路－3.5m以上',
                      '^3$':'単路－5.5m以上',
                      '^4$':'単路－9.0m以上',
                      '^5$':'単路－13.0m以上',
                      '^6$':'単路－19.5m以上',
                      '^11$':'交差点－小（5.5m未満）－小',
                      '^14$':'交差点－中（5.5m以上）－小',
                      '^15$':'交差点－中（5.5m以上）－中',
                      '^17$':'交差点－大（13.0ｍ以上）－小',
                      '^18$':'交差点－大（13.0ｍ以上）－中',
                      '^19$':'交差点－大（13.0ｍ以上）－大',                
                      '^0$':'一般交通の場所',
                        }

traffic_accident_data['road_width_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['車道幅員'].astype(str)))


# 道路線形
replace_list = {'^1$':'カーブ・屈折－右－上り',
                      '^2$':'カーブ・屈折－右－下り',
                      '^3$':'カーブ・屈折－右－平坦',
                      '^4$':'カーブ・屈折－左－上り',
                      '^5$':'カーブ・屈折－左－下り',
                      '^6$':'カーブ・屈折－左－平坦',
                      '^7$':'直線－上り',
                      '^8$':'直線－下り',
                      '^9$':'直線－平坦',
                      '^0$':'一般交通の場所',
                        }

traffic_accident_data['road_alignment_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['道路線形'].astype(str)))


# 衝突地点
replace_list = {'^1$':'単路（交差点付近を含む）',
                      '^30$':'交差点内',
                      '^20$':'その他',
                        }

traffic_accident_data['road_alignment_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['衝突地点'].astype(str)))


# ゾーン規制
replace_list = {'^1$':'ゾーン30',
                      '^70$':'規制なし',
                        }

traffic_accident_data['zone_regulation_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['ゾーン規制'].astype(str)))


# 中央分離帯施設等
replace_list = {'^1$':'中央分離帯',
                      '^2$':'中央線－高輝度標示',
                      '^3$':'中央線－チャッターバー等',
                      '^6$':'中央線－ポストコーン', 
                      '^4$':'中央線－ペイント',
                      '^5$':'中央分離なし',
                      '^0$':'一般交通の場所',                
                        }

traffic_accident_data['zone_regulation_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                traffic_accident_data['中央分離帯施設等'].astype(str)))


# 歩車道区分
replace_list = {'^1$':'区分あり－防護柵等',
                      '^2$':'区分あり－縁石・ブロック等',
                      '^3$':'区分あり－路側帯',
                      '^4$':'区分なし', 
                        }

traffic_accident_data['pedestrian_road_division_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['歩車道区分'].astype(str)))


# 事故類型
replace_list = {'^1$':'人対車両',
                      '^21$':'車両相互',
                      '^41$':'車両単独',
                      '^61$':'列車', 
                        }

traffic_accident_data['accident_vehicle_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['事故類型'].astype(str)))


# 年齢
replace_list = {'^1$':'0～24歳',
                      '^25$':'25～34歳',
                      '^35$':'35～44歳',
                      '^45$':'45～54歳', 
                      '^55$':'55～64歳',
                      '^65$':'65～74歳', 
                      '^75$':'75歳以上',
                      '^0$':'不明',                
                        }

traffic_accident_data['age_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['年齢（当事者A）'].astype(str)))
traffic_accident_data['age_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['年齢（当事者B）'].astype(str)))


# 当事者種別
replace_list = {'^1$':'乗用車－大型車',
                      '^2$':'乗用車－中型車',
                      '^7$':'乗用車－準中型車',
                      '^3$':'乗用車－普通車',
                      '^4$':'乗用車－軽自動車',
                      '^5$':'乗用車－ミニカー',
                      '^11$':'貨物車－大型車',
                      '^12$':'貨物車－中型車',
                      '^17$':'貨物車－準中型車',
                      '^13$':'貨物車－普通車',
                      '^14$':'貨物車－軽自動車',
                      '^21$':'特殊車－大型－農耕作業用',
                      '^22$':'特殊車－大型－その他',
                      '^23$':'特殊車－小型－農耕作業用',
                      '^24$':'特殊車－小型－その他',
                      '^31$':'二輪車－自動二輪－小型二輪－751ｃｃ以上', 
                      '^32$':'二輪車－自動二輪－小型二輪－401～750ｃｃ',
                      '^33$':'二輪車－自動二輪－小型二輪－251～400cc',
                      '^34$':'二輪車－自動二輪－軽二輪－126～250cc',
                      '^35$':'二輪車－自動二輪－原付二種－51～125cc', 
                      '^36$':'二輪車－原付自転車',
                      '^41$':'路面電車',
                      '^42$':'列車',
                      '^51$':'軽車両－自転車',
                      '^52$':'軽車両－駆動補助機付自転車',
                      '^59$':'軽車両－その他',
                      '^61$':'歩行者',
                      '^71$':'歩行者以外の道路上の人（補充票のみ）',
                      '^72$':'道路外の人（補充票のみ）',
                      '^75$':'物件等',
                      '^76$':'相手なし',
                      '^0$':'対象外当事者',                
                        }

traffic_accident_data['parties_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['当事者種別（当事者A）'].astype(str)))
traffic_accident_data['parties_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['当事者種別（当事者B）'].astype(str)))


# 用途別
replace_list = {'^1$':'事業用',
                      '^31$':'自家用',
                      '^0$':'対象外当事者',
                      '^$':'ー', 
                        }

traffic_accident_data['use_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['用途別（当事者A）'].astype(str)))
traffic_accident_data['use_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['用途別（当事者B）'].astype(str)))

# 車両形状
replace_list = {'^1$':'乗用車',
                      '^11$':'貨物車',
                      '^0$':'対象外当事者',
                      '^$':'ー', 
                        }

traffic_accident_data['vehicle_shape_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['車両形状（当事者A）'].astype(str)))
traffic_accident_data['vehicle_shape_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['車両形状（当事者B）'].astype(str)))

# 速度規制（指定のみ）
replace_list = {'^1$':'20㎞／ｈ以下',
                      '^2$':'30㎞／ｈ以下',
                      '^3$':'40㎞／ｈ以下',
                      '^4$':'50㎞／ｈ以下',
                      '^5$':'60㎞／ｈ以下',
                      '^6$':'70㎞／ｈ以下',
                      '^7$':'80㎞／ｈ以下',
                      '^8$':'100㎞/h以下',
                      '^9$':'100㎞/h超過',
                      '^10$':'指定の速度規制なし等',
                      '^0$':'対象外当事者',                
                        }

traffic_accident_data['speed_regulation_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['速度規制（指定のみ）（当事者A）'].astype(str)))
traffic_accident_data['speed_regulation_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['速度規制（指定のみ）（当事者B）'].astype(str)))

# 車両の衝突部位
replace_list = {'^1':'前_中央_',
                      '^2':'右_中央_',
                      '^3':'後_中央_',
                      '^4':'左_中央_',
                      '^5':'前_右_',
                      '^6':'後_右_',
                      '^7':'後_左_',
                      '^8':'前_左_',
                      '^0':'それ以外_',                
                        }

traffic_accident_data['collision_site_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['車両の衝突部位（当事者A）'].astype(str)))
traffic_accident_data['collision_site_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['車両の衝突部位（当事者B）'].astype(str)))
replace_list = {'1$':'前_中央',
                      '2$':'右_中央',
                      '3$':'後_中央',
                      '4$':'左_中央',
                      '5$':'前_右',
                      '6$':'後_右',
                      '7$':'後_左',
                      '8$':'前_左',
                      '0$':'それ以外',                
                        }

traffic_accident_data['collision_site_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['collision_site_type_a'].astype(str)))
traffic_accident_data['collision_site_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['collision_site_type_b'].astype(str)))

# 車両の損壊程度
replace_list = {'^1$':'大破',
                      '^2$':'中破',
                      '^3$':'小破',
                      '^4$':'損壊なし',
                      '^0$':'対象外当事者',
                      '^$':'ー',   
                        }

traffic_accident_data['damage_to_vehicle_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['車両の損壊程度（当事者A）'].astype(str)))
traffic_accident_data['damage_to_vehicle_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['車両の損壊程度（当事者B）'].astype(str)))


# エアバッグの装備
replace_list = {'^1$':'装備あり作動',
                      '^2$':'その他',
                      '^0$':'対象外当事者',
                        }

traffic_accident_data['airbag_equipment_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['エアバッグの装備（当事者A）'].astype(str)))
traffic_accident_data['airbag_equipment_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['エアバッグの装備（当事者B）'].astype(str)))

# サイドエアバッグの装備
replace_list = {'^1$':'装備あり作動',
                      '^2$':'その他',
                      '^0$':'対象外当事者',
                        }

traffic_accident_data['side_airbag_equipment_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['サイドエアバッグの装備（当事者A）'].astype(str)))
traffic_accident_data['side_airbag_equipment_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['サイドエアバッグの装備（当事者B）'].astype(str)))


# 人身損傷程度
replace_list = {'^1$':'死亡',
                      '^2$':'負傷',
                      '^4$':'損傷なし',                
                      '^0$':'対象外当事者',
                        }

traffic_accident_data['personal_injury_type_a'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['人身損傷程度（当事者A）'].astype(str)))
traffic_accident_data['personal_injury_type_b'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['人身損傷程度（当事者B）'].astype(str)))


# 曜日(発生年月日)
replace_list = {'^1$':'日',
                      '^2$':'月',
                      '^3$':'火',
                      '^4$':'水', 
                      '^5$':'木', 
                      '^6$':'金',                 
                      '^7$':'土',                 
                        }

traffic_accident_data['weekday_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['曜日(発生年月日)'].astype(str)))


# 祝日
replace_list = {'^1$':'当日',
                      '^2$':'前日',
                      '^3$':'その他',
                        }

traffic_accident_data['holiday_type'] = list(map(lambda text:multiple_replace(text, replace_list) ,
                                                                                            traffic_accident_data['祝日(発生年月日)'].astype(str)))

# 教師データ
traffic_accident_data["death_flag"] = np.where(traffic_accident_data["accident_type"] == "死亡", 1, 0)

# 緯度経度が変なデータを除外
traffic_accident_data = traffic_accident_data[traffic_accident_data["地点\u3000緯度（北緯）"] > 100000]
traffic_accident_data = traffic_accident_data[traffic_accident_data["地点\u3000経度（東経）"] > 100000].reset_index(drop=True)

traffic_accident_data["latitude"] = list(map(lambda id:dms2deg(traffic_accident_data["地点\u3000緯度（北緯）"][id]),
                                                                                     range(traffic_accident_data.index.size)))
traffic_accident_data["longitude"] = list(map(lambda id:dms2deg(traffic_accident_data["地点\u3000経度（東経）"][id]),
                                                                                     range(traffic_accident_data.index.size)))
traffic_accident_data["latitude"] = traffic_accident_data["latitude"].astype(float)
traffic_accident_data["longitude"] = traffic_accident_data["longitude"].astype(float)

# メッシュデータの付与
traffic_accident_data["third_mesh"] = list(map(lambda id:latlon2mesh(traffic_accident_data.latitude[id],
                                                                traffic_accident_data.longitude[id]), range(traffic_accident_data.index.size)))


# データの保存
traffic_accident_data.to_pickle("../data/traffic_accident_data.pickle")
