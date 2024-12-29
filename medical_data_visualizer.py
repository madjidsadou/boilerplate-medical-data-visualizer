import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')
print(df.columns)
# 2
df['bmi'] = df['weight']/((df['height']/100)**2)
df['overweight'] = (df['bmi']>25).astype(int)
df = df.drop(columns=['bmi'])

# 3
df['gluc'] = (df['gluc']>1).astype(int)
df['cholesterol'] = (df['cholesterol']>1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
    df, 
    id_vars=['cardio'],  # Keep 'cardio' as an identifier column
    value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],  # Columns to melt
    var_name='feature',  # Name of the new column for feature names
    value_name='value'  # Name of the new column for feature values
)

    # 6
    df_cat = df_cat.groupby(['cardio', 'feature', 'value']).size().reset_index(name='count')
    df_cat = df_cat.rename(columns={'feature': 'category'})


    # 8
    sns.catplot(
    data=df_cat, 
    x='category', 
    hue='value', 
    col='cardio', 
    kind='bar', 
    y='count', 
    height=5, 
    aspect=1.5
)

# Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    fig =     sns.catplot(
    data=df_cat, 
    x='category', 
    hue='value', 
    col='cardio', 
    kind='bar', 
    y='count', 
    height=5, 
    aspect=1.5
    )

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.copy()
    df_heat = df_heat.drop(df_heat[(df_heat['ap_lo'] >= df_heat['ap_hi'])].index)  
    df_heat = df_heat.drop(df_heat[(df_heat['height'] < df_heat['height'].quantile(0.025))].index)
    df_heat = df_heat.drop(df_heat[(df_heat['height'] > df_heat['height'].quantile(0.975))].index)
    df_heat = df_heat.drop(df_heat[(df_heat['weight'] < df_heat['weight'].quantile(0.025))].index)  
    df_heat = df_heat.drop(df_heat[(df['weight'] > df_heat['weight'].quantile(0.975))].index) 

    corr = df_heat.corr()

    corr = corr.round(1)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))  

    # Draw the heatmap
    sns.heatmap(
        corr,
        center=0,
        square=True,
        annot=True,
        linewidths=3,
        mask=mask,
        cbar_kws={'shrink': 0.8},  
        ax=ax
    )

    # Save the figure
    fig.savefig('heatmap.png')

    plt.close(fig)  
    return fig

