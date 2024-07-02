import pandas as pd
import plotly.graph_objects as go
import re
from constants import CITY_ABBR
# import numpy as np

def draw_radar(target:str, *data:pd.DataFrame):
    """
    Draw Radar chart for multiple cities
    
    Args:
        target: 'coverage' or 'improvement'
        data: multiple cities' data
    """
    fig = go.Figure()

    for d in data:
        fig.add_trace(go.Scatterpolar(
            r=d[target],
            theta=d['goal_text'],
            fill='toself',
            name=d['cityname'].unique()[0],
            line=dict(
                color=d['color'].unique()[0],
                width=2
            )
        ))

    # Add a circle with 0 value to make the radar chart more clear
    circle_list = d['goal_text'].tolist()
    circle_list.append(circle_list[0])

    fig.add_trace(go.Scatterpolar(
        mode='lines',
        r=[0]*18,
        theta=circle_list,
        fillcolor=None,
        name= 'Zero Line',
        line=dict(
            color='white',
            width=1
        )
    ))

    data_range = [0,1] if target=='coverage' else [-1, 1]
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=data_range
            ), 
            angularaxis=dict(
            direction = 'clockwise',
            rotation=90,
            )   
        ),
        showlegend=True,
        title=f"SDG Goals \"{target}\" Radar Chart",
        width=800,
        height=500,
        # style
        template='plotly_dark',
    )

    fig.show()

def draw_bar(target:str, *data:pd.DataFrame):
    """
    Draw bar chart for UN SDGs coverage

    Args:
        data: multiple cities' data
    """

    # Sort the data by un_coverage(descending order) 
    data = sorted(data, key=lambda x: float(re.search(r'\((\d+\.\d+)%\)', x.loc[0, target]).group(1)), reverse=False)

    fig = go.Figure()
    
    for d in data:
        # extract 37.1% from 92/248 (37.1%) 
        match = re.search(r'\((\d+\.\d+)%\)', d[target][0])
        target_val = float(match.group(1))

        fig.add_trace(go.Bar(
            x=[target_val],
            y=d.city_name,
            # name=d.city_name,
            # marker_color=d.color,
            orientation='h',
            text=d[target],
            marker_color=d.color,
        ))
    
    fig.update_layout(
        title=f'{target}',
        xaxis=dict(
            title='Percentage',
            range=[0, 100]
        ),
        yaxis=dict(
            title='City'
        ),
        barmode='group',
        showlegend=False,
        width=800,
        height=300,
        template='plotly_dark',
    )

    fig.show()

def draw_sankey_one_city(data:pd.DataFrame, city_name:str ,sdg_target:str = 'SDG11'):
    """
    Args:
        data: pd.DataFrame
        sdg_target: str, ex: 'SDG11'
    """
    selection = data[data['goal']==sdg_target].sort_values(by='source', ascending=True)
    node_labels = list(selection['source'].unique()) + list(selection['target'].unique())
    node_customdata = list(selection['un_indicator'].unique()) + list(selection['city_indicator'].unique())
    # 修正：node to node
    # link_customdata = list(selection['value'])

    fig = go.Figure()
    fig.add_trace(go.Sankey(
        node=dict(
            pad=5,
            thickness=20,
            # line= dict(color="black", width=0.5),
            label = node_labels,
            hoverinfo='all',
            customdata=node_customdata,
            hovertemplate='%{label}<br> %{customdata}'
        ),
        link=dict(
            source=[node_labels.index(source) for source in selection['source']],
            target=[node_labels.index(target) for target in selection['target']],
            value=selection['value'],
            # color='blue',
            # customdata = link_customdata,
            hoverinfo='all',
            hovertemplate='source: %{source.label} <br> target: %{target.label} <br> value: %{value} <br> %{customdata}'
        )
    ))

    fig.update_layout(
        title_text=f"Sankey Diagram of {sdg_target} in {city_name}", # 標題
        margin=dict(l=30, r=30, t=60, b=30), # 圖表邊界
        width=800,
        height=400,
        # style
        template='plotly_dark',
    )
    fig.show()

def draw_line(data, city_name:str, sdg_target:str = 'SDG11'):
    """
    Draw line chart for multiple cities' improvement

    Args:
        data: multiple cities' data
    """
    selection = (data[(data["goal"]==sdg_target) & (data['target']!='Nothing')]
                 .drop_duplicates(subset=['target'], keep='first'))

    fig = go.Figure()
    
    for _, d in selection.iterrows():
        fig.add_trace(go.Scatter(
            x=[2018, 2019, 2020], 
            y=[d['2018_rate'], d['2019_rate'], d['2020_rate']],
            mode='lines+markers', 
            name=d['id'],
            hoverinfo='all',
            customdata=[d['city_indicator']]*3, 
            # 增加值 各個值的標籤

            hovertemplate='%{x} improvement %{y} <br> %{customdata} <br> mapped'

        ))
    
    fig.update_layout(
        # title_text=f"{sdg_target}<br>Sankey Diagram of {sdg_target} in {city_name}", # 標題
        xaxis=dict(
            tickmode='array',
            tickvals=[2018, 2019, 2020],
        ),
        template='plotly_dark',
        width=800,
        height=200,
        margin=dict(l=30, r=30, t=30, b=30), # 圖表邊界
    )

    fig.show()

def draw_sankey_multi_cities(main_data, *other_data, sdg_target:str, indicator_target:str = None):
    """
    Args:
        main_data: pd.DataFrame, 主要城市的資料, 在sankey圖的左邊
        other_data: pd.DataFrame, 其他城市的資料, 在sankey圖的右邊
        sdg_target: str, ex: 'SDG11' (required, default: 'SDG11')
        indicator_target: str, ex: '1-1-1' (optional, default: None)
    """

    # 加上city_name縮寫，避免多城市搞混 ex: NTP01
    main_data = main_data.copy()
    main_city = main_data['city_name'].unique()[0]
    main_data['target'] = ' (' + main_data['city_name'].map(CITY_ABBR) + ')' + main_data['target']

    # main_data source 和 target 互換
    main_data['source'], main_data['target'] = main_data['target'], main_data['source']
    selection = main_data[main_data['goal']==sdg_target].sort_values(by='target', ascending=True)


    other_city = []
    for data in other_data:
        data = data.copy()
        other_city.append(data['city_name'].unique()[0])
        # 加上city_name縮寫，避免多城市搞混 ex: TP01, TY01
        data['target'] = ' (' + CITY_ABBR[data['city_name'].unique()[0]] + ')' + data['target']

        if indicator_target:
            data = data[data['source']==f"Indicator{indicator_target}"]
        selection = pd.concat([selection, data[(data['goal']==sdg_target) & (data['target'] != ' (' + CITY_ABBR[data['city_name'].unique()[0]] + ')'+'Nothing')].sort_values(by='source', ascending=True)])

    node_ids = list(selection['source'].unique()) + list(selection['target'].unique())
    
    def node_id_to_label(node_id):
        """
        get node_id's indicator title for nodes' hover template
        """
        if "Indicator" in node_id:
            return selection.loc[(selection['source'] == node_id) | (selection['target'] == node_id), 'un_indicator'].values[0]
        else:
            return selection.loc[(selection['source'] == node_id) | (selection['target'] == node_id), 'city_indicator'].values[0]

    node_customdata = [node_id_to_label(node_id) for node_id in node_ids]
    # 修正：node to node
    # link_customdata = list(selection['value'])

    fig = go.Figure()
    fig.add_trace(go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            # line= dict(color="black", width=0.5),
            label = node_ids,
            hoverinfo='all',
            customdata=node_customdata,
            hovertemplate='%{label}<br> %{customdata}',
            # align='left' if indicator_target else 'justify',
            align='left',
        ),
        link=dict(
            source=[node_ids.index(source) for source in selection['source']],
            target=[node_ids.index(target) for target in selection['target']],
            value=selection['value'],
            # color='blue',
            # customdata = link_customdata,
            # color='blue',
            # customdata = link_customdata,
            hoverinfo='all',
            hovertemplate='source: %{source.label} <br> target: %{target.label} <br> value: %{value} <br> %{customdata}'
        )
    ))

    fig.update_layout(
        title_text=f"Sankey Diagram of {sdg_target} in {main_city} vs {', '.join(other_city)}", # 標題
        margin=dict(l=30, r=30, t=60, b=30), # 圖表邊界
        width=800,
        height=400,
        # style
        # style
        template='plotly_dark',
    )
    fig.show()


def draw_sankey_open_data(main_data, other_data, sdg_target:str, indicator_target:str = None, threshold = 0.55):
    """
    Args:
        main_data: pd.DataFrame, 主要城市的資料, 在sankey圖的左邊
        other_data: pd.DataFrame, 新北市開放資料
        sdg_target: str, ex: 'SDG11' (required, default: 'SDG11')
        indicator_target: str, ex: '1-1-1' (optional, default: None)
        threshold: float, 門檻值, 用來過濾數值小於門檻值的資料 (optional, default: 0.55)
    """

    # 加上city_name縮寫，避免多城市搞混 ex: NTP01
    main_data = main_data.copy()
    # main_city = main_data['city_name'].unique()[0]
    main_data.loc[main_data['target']!='Nothing', 'target'] = main_data['target'] + ' (' + main_data['department'] + ')' # 加上主責單位

    # main_data source 和 target 互換
    main_data['source'], main_data['target'] = main_data['target'], main_data['source']
    selection = main_data[main_data['goal']==sdg_target].sort_values(by='target', ascending=True)


    # other_city = []
    # for data in other_data:
    data = other_data.copy()
    # other_city.append(data['city_name'].unique()[0])
    # 加上city_name縮寫，避免多城市搞混 ex: TP01, TY01
    # data['target'] = ' (' + CITY_ABBR[data['city_name'].unique()[0]] + ')' + data['target']

    grouped = (data[(data['goal']==sdg_target) & (data['target']!='Nothing') & (data['value']> threshold)]
        .groupby(['source','target'])) 

    data = grouped.first().reset_index()
    data = pd.concat([data, grouped['city_indicator'].agg([list, 'count']).reset_index().drop(['source', 'target'], axis=1)], axis=1)

    if indicator_target:
        data = data[data['source']==f"Indicator{indicator_target}"]
    selection = pd.concat([selection, data])
    

    node_ids = list(selection['source'].unique()) + list(selection['target'].unique())
    
    def node_id_to_label(node_id):
        """
        get node_id's indicator title for nodes' hover template
        """
        if "Indicator" in node_id:
            return selection.loc[(selection['source'] == node_id) | (selection['target'] == node_id), 'un_indicator'].values[0]
        else:
            return selection.loc[(selection['source'] == node_id) | (selection['target'] == node_id), 'city_indicator'].values[0]

    node_customdata = [node_id_to_label(node_id) for node_id in node_ids]
    # print(node_customdata)
    # 修正：node to node
    # link_customdata = list(selection['value'])

    fig = go.Figure()
    fig.add_trace(go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            # line= dict(color="black", width=0.5),
            label = node_ids,
            hoverinfo='all',
            customdata=node_customdata,
            hovertemplate='%{label}<br> %{customdata}',
            # align='left' if indicator_target else 'justify',
            align='left',
        ),
        link=dict(
            source=[node_ids.index(source) for source in selection['source']],
            target=[node_ids.index(target) for target in selection['target']],
            value=selection['value'],
            # color='blue',
            # customdata = link_customdata,
            # color='blue',
            # customdata = link_customdata,
            hoverinfo='all',
            hovertemplate='source: %{source.label} <br> target: %{target.label} <br> value: %{value} <br> %{customdata}'
        )
    ))

    fig.update_layout(
        # title_text=f"Sankey Diagram of {sdg_target} in {main_city} vs {', '.join(other_city)}", # 標題
        margin=dict(l=30, r=30, t=60, b=30), # 圖表邊界
        width=800,
        height=400,
        # style
        template='plotly_dark',
    )
    fig.show()