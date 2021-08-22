import pandas as pd
import numpy as np


traffic_accident_data = pd.read_pickle("../data/traffic_accident_data.pickle")

# テスト用
test_df = traffic_accident_data[traffic_accident_data.accident_date > "2020-12-01"].reset_index(drop=True)

# メッシュごとの特徴量作成用
mesh_df = traffic_accident_data[(traffic_accident_data.accident_date > "2019-01-01") &\
                                                   (traffic_accident_data.accident_date < "2019-09-01")].reset_index(drop=True)

# 学習用
train_df = traffic_accident_data[(traffic_accident_data.accident_date >= "2019-09-01") &\
                                                   (traffic_accident_data.accident_date <= "2020-12-01")].reset_index(drop=True)


mesh_summary_df = mesh_df.groupby('third_mesh')['death_flag'].agg([np.sum, "count"]).reset_index()
mesh_summary_df.columns = ["third_mesh", "death_count", "accident_count"]

summary_column_name_list = ["road_type","road_bypass","road_updown_type","day_night_type","weather_type",
                                                "terrain_type","road_condition_type","road_shape_type","traffic_lights_type",
                                                "pause_sign_type_a","pause_sign_type_b","pause_display_type_a","pause_display_type_b",
                                                "road_width_type","road_alignment_type","zone_regulation_type",
                                                "pedestrian_road_division_type","accident_vehicle_type","age_type_a","age_type_b",
                                                "parties_type_a","parties_type_b","use_type_a","use_type_b","vehicle_shape_type_a",
                                                "vehicle_shape_type_b","speed_regulation_type_a","speed_regulation_type_b",
                                                "collision_site_type_a","collision_site_type_b","damage_to_vehicle_type_a",
                                                "damage_to_vehicle_type_b","airbag_equipment_type_a","airbag_equipment_type_b",
                                                "side_airbag_equipment_type_a","side_airbag_equipment_type_b","weekday_type","holiday_type"]

for each_variable_name in summary_column_name_list:
    df_pv = pd.pivot_table(data=mesh_df,
                                       fill_value=0,
                                       index="third_mesh",
                                       columns=each_variable_name,
                                       aggfunc = {each_variable_name:"count"}).reset_index()
    column_list = [df_pv.columns.levels[0][1]]
    add_column_list = [ each_variable_name + "_"+ str(_) for _ in df_pv.columns.levels[1].tolist()[:-1]]
    column_list.extend(add_column_list)
    df_pv.columns = column_list
    mesh_summary_df = pd.merge(mesh_summary_df, df_pv, on="third_mesh", how="left")


teacher_flag = train_df.death_flag

train_columns = ["weekday_type", "holiday_type", "pref_name", "road_type",
                          "road_bypass", "road_updown_type", "day_night_type",
                          "weather_type", "terrain_type", "road_condition_type",
                          "road_shape_type", "traffic_lights_type", "pause_sign_type_a",
                          "pause_display_type_a", "road_width_type", "road_alignment_type",
                          "zone_regulation_type", "pedestrian_road_division_type",
                          "speed_regulation_type_a", "third_mesh"]

train_df = train_df[train_columns]

# 一気にカテゴリ化
cols = train_columns[:-1]
train_df[cols] = train_df[cols].astype('category')
 
# ダミー変数を一気に作る
for each_columns in cols:
    dummy_df = pd.get_dummies( train_df[each_columns], drop_first= False)
    dummy_df.columns =  ["dummy_" + each_columns + "_" + str(i) for i in dummy_df.columns.tolist()]
    
    train_df = pd.concat([train_df,
                          dummy_df], axis=1)


dataset = pd.concat([train_df["third_mesh"],
                                train_df.filter(regex="dummy_weekday_type"),
                                train_df.filter(regex="dummy_holiday_type"),
                                train_df.filter(regex="dummy_pref_name"),
                                train_df.filter(regex="dummy_road_type"),
                                train_df.filter(regex="dummy_road_bypass"),
                                train_df.filter(regex="dummy_road_updown_type"),
                                train_df.filter(regex="dummy_day_night_type"),
                                train_df.filter(regex="dummy_weather_type"),
                                train_df.filter(regex="dummy_terrain_type"),
                                train_df.filter(regex="dummy_road_condition_type"),
                                train_df.filter(regex="dummy_road_shape_type"),
                                train_df.filter(regex="dummy_traffic_lights_type"),
                                train_df.filter(regex="dummy_pause_sign_type_a"),
                                train_df.filter(regex="dummy_pause_display_type_a"),
                                train_df.filter(regex="dummy_road_width_type"),
                                train_df.filter(regex="dummy_road_alignment_type"),
                                train_df.filter(regex="dummy_zone_regulation_type"),
                                train_df.filter(regex="dummy_pedestrian_road_division_type"),
                                train_df.filter(regex="dummy_speed_regulation_type_a"),
                      ], axis=1)

dataset = pd.merge(dataset, mesh_summary_df, on="third_mesh", how="left")

dataset.to_pickle("../data/dataset.pickle")
teacher_flag.to_pickle("../data/teacher_flag.pickle")