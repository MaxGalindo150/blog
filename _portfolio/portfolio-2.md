---
title: "Analysis of Contaminant Concentration Trends in Mexico City (2015-2023) (CODE)"
excerpt: "Code for analysis of RAMA data base"
collection: portfolio
---

The following code facilitated the analysis presented in the blog post titled *Analysis of Contaminant Concentration Trends in Mexico City (2015-2023)*. For a detailed view of this analysis, feel free to visit the [blog post section](http://maxgalindo.sytes.net/year-archive/).

## Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
```

Data for each pollutant from year 2015 to 2023 is stored in directories labeled accordingly. The pollutants included are:

- `CO`: Carbon Monoxide
- `NO`: Nitric Oxide
- `NO2`: Nitrogen Dioxide
- `NOX`: Nitrogen Oxides
- `O3`: Ozone
- `PM10`: Particulate Matter less than 10 micrometers
- `PM25`: Particulate Matter less than 2.5 micrometers
- `PMCO`: Coarse Particulate Matter
- `SO2`: Sulfur Dioxide


We load the data frame and aggregate each pollutant's records across every year from 2015 to 2023.

```python
contaminants = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM25', 'PMCO', 'SO2']
years = ['15', '16', '17', '18', '19', '20', '21', '22', '23']
```


```python

for contaminant in contaminants:
    contaminant_per_year = []
    for year in years:
        print(year, contaminant)
        df = pd.read_excel(f'./{year}/20{year}{contaminant}.xls')
        df['Year'] = f'20{year}'
        contaminant_per_year.append(df)
    contaminant_per_year = pd.concat(contaminant_per_year)
    contaminant_per_year = contaminant_per_year.replace(np.nan, -99)
    contaminant_per_year = contaminant_per_year.replace(-99, np.nan)
    contaminant_per_year.to_csv(f'./{contaminant}.csv', index=False)
```



After merging each pollutant with its corresponding data from each year, we load each of these DataFrames into a dictionary for easier manipulation.


```python
contaminants_data = {}
for contaminant in contaminants:
    contaminants_data[contaminant] = pd.read_csv(f'./{contaminant}.csv')
```


```python
# Display the first 3 records for each pollutant
for contaminant in contaminants:
    print("Contaminant: ", contaminant)
    print(contaminants_data[contaminant].head(3))
    print("="*100)
```

    Contaminant:  CO
            FECHA  HORA  ACO  AJM  ATI  BJU  CAM  CCA  CHO  CUA  ...  TLI  UAX  \
    0  2015-01-01     1  0.6  1.2  1.1  NaN  1.2  1.7  NaN  1.7  ...  1.1  2.3   
    1  2015-01-01     2  0.8  1.4  1.4  NaN  1.3  1.5  NaN  2.1  ...  1.1  2.1   
    2  2015-01-01     3  0.8  1.3  1.4  NaN  1.3  1.6  NaN  1.6  ...  1.5  1.4   
    
       UIZ  VIF  XAL  Year  MPA  CUT  FAR  SAC  
    0  1.4  1.0  2.8  2015  NaN  NaN  NaN  NaN  
    1  1.2  1.1  3.4  2015  NaN  NaN  NaN  NaN  
    2  1.4  1.5  2.8  2015  NaN  NaN  NaN  NaN  
    
    [3 rows x 37 columns]
    ====================================================================================================
    Contaminant:  NO
            FECHA  HORA   ACO   AJM  ATI  BJU  CAM   CCA  CHO  COY  ...  TPN  \
    0  2015-01-01     1  14.0   3.0  3.0  NaN  NaN  11.0  NaN  NaN  ...  3.0   
    1  2015-01-01     2  29.0  13.0  5.0  NaN  NaN  11.0  NaN  NaN  ...  4.0   
    2  2015-01-01     3  37.0   8.0  9.0  NaN  NaN  20.0  NaN  NaN  ...  2.0   
    
        UAX   UIZ   VIF    XAL  Year  MPA  FAR  SAC  AJU  
    0  43.0  45.0  33.0  113.0  2015  NaN  NaN  NaN  NaN  
    1  50.0  43.0  38.0  149.0  2015  NaN  NaN  NaN  NaN  
    2  42.0  47.0  36.0  115.0  2015  NaN  NaN  NaN  NaN  
    
    [3 rows x 40 columns]
    ====================================================================================================
    Contaminant:  NO2
            FECHA  HORA   ACO   AJM   ATI  BJU  CAM   CCA  CHO  COY  ...   UAX  \
    0  2015-01-01     1  21.0  58.0  30.0  NaN  NaN  52.0  NaN  NaN  ...  55.0   
    1  2015-01-01     2  21.0  58.0  35.0  NaN  NaN  44.0  NaN  NaN  ...  41.0   
    2  2015-01-01     3  21.0  52.0  33.0  NaN  NaN  44.0  NaN  NaN  ...  28.0   
    
        UIZ   VIF   XAL  Year  MPA  FAR  GAM  SAC  AJU  
    0  32.0  29.0  38.0  2015  NaN  NaN  NaN  NaN  NaN  
    1  28.0  28.0  34.0  2015  NaN  NaN  NaN  NaN  NaN  
    2  29.0  27.0  29.0  2015  NaN  NaN  NaN  NaN  NaN  
    
    [3 rows x 41 columns]
    ====================================================================================================
    Contaminant:  NOX
            FECHA  HORA   ACO   AJM   ATI  BJU  CAM   CCA  CHO  COY  ...   TPN  \
    0  2015-01-01     1  35.0  61.0  32.0  NaN  NaN  63.0  NaN  NaN  ...  24.0   
    1  2015-01-01     2  50.0  71.0  40.0  NaN  NaN  55.0  NaN  NaN  ...  27.0   
    2  2015-01-01     3  58.0  59.0  42.0  NaN  NaN  64.0  NaN  NaN  ...  18.0   
    
        UAX   UIZ   VIF    XAL  Year  MPA  FAR  SAC  AJU  
    0  98.0  78.0  62.0  150.0  2015  NaN  NaN  NaN  NaN  
    1  90.0  71.0  66.0  183.0  2015  NaN  NaN  NaN  NaN  
    2  71.0  76.0  63.0  144.0  2015  NaN  NaN  NaN  NaN  
    
    [3 rows x 40 columns]
    ====================================================================================================
    Contaminant:  O3
            FECHA  HORA  ACO  AJM   AJU   ATI  BJU  CAM  CCA  CHO  ...   TPN  UAX  \
    0  2015-01-01     1  4.0  8.0  20.0  21.0  NaN  2.0  2.0  3.0  ...  20.0  2.0   
    1  2015-01-01     2  5.0  2.0  31.0  16.0  NaN  2.0  4.0  6.0  ...  15.0  2.0   
    2  2015-01-01     3  6.0  3.0  26.0  10.0  NaN  3.0  1.0  4.0  ...  21.0  2.0   
    
       UIZ  VIF  XAL  Year  MPA  FAR  LAA  SAC  
    0  2.0  2.0  NaN  2015  NaN  NaN  NaN  NaN  
    1  2.0  3.0  NaN  2015  NaN  NaN  NaN  NaN  
    2  4.0  3.0  NaN  2015  NaN  NaN  NaN  NaN  
    
    [3 rows x 42 columns]
    ====================================================================================================
    Contaminant:  PM10
            FECHA  HORA    ACO  AJM    ATI  BJU    CAM    CHO  CUA    CUT  ...  \
    0  2015-01-01     1   84.0  NaN  122.0  NaN   95.0   86.0  NaN  186.0  ...   
    1  2015-01-01     2  110.0  NaN  179.0  NaN  135.0   88.0  NaN  177.0  ...   
    2  2015-01-01     3  140.0  NaN  153.0  NaN  166.0  120.0  NaN  223.0  ...   
    
         VIF  XAL  Year  MPA  LLA  LPR  FAR  GAM  SAC  NEZ  
    0  211.0  NaN  2015  NaN  NaN  NaN  NaN  NaN  NaN  NaN  
    1  246.0  NaN  2015  NaN  NaN  NaN  NaN  NaN  NaN  NaN  
    2  183.0  NaN  2015  NaN  NaN  NaN  NaN  NaN  NaN  NaN  
    
    [3 rows x 34 columns]
    ====================================================================================================
    Contaminant:  PM25
            FECHA  HORA  AJM  AJU  BJU    CAM    CCA    COY  GAM    HGM  ...  \
    0  2015-01-01     1  NaN  NaN  NaN   80.0  118.0  159.0  NaN   85.0  ...   
    1  2015-01-01     2  NaN  NaN  NaN  118.0  107.0  177.0  NaN  104.0  ...   
    2  2015-01-01     3  NaN  NaN  NaN  148.0  121.0  171.0  NaN  112.0  ...   
    
         UIZ  XAL  Year  MPA  CUA  CUT  FAC  MON  FAR  SAC  
    0   81.0  NaN  2015  NaN  NaN  NaN  NaN  NaN  NaN  NaN  
    1  104.0  NaN  2015  NaN  NaN  NaN  NaN  NaN  NaN  NaN  
    2  135.0  NaN  2015  NaN  NaN  NaN  NaN  NaN  NaN  NaN  
    
    [3 rows x 30 columns]
    ====================================================================================================
    Contaminant:  PMCO
            FECHA  HORA  AJM  BJU   CAM   HGM  INN   MER   MGH   PED   SAG   SFE  \
    0  2015-01-01     1  NaN  NaN  15.0  32.0  NaN  38.0  19.0  36.0  35.0  33.0   
    1  2015-01-01     2  NaN  NaN  17.0  35.0  NaN  34.0  26.0  38.0  44.0  31.0   
    2  2015-01-01     3  NaN  NaN  18.0  33.0  NaN  31.0  23.0  36.0  33.0  30.0   
    
       TLA   UIZ  XAL  Year  MPA  FAR  GAM  SAC  
    0  NaN  35.0  NaN  2015  NaN  NaN  NaN  NaN  
    1  NaN  34.0  NaN  2015  NaN  NaN  NaN  NaN  
    2  NaN  41.0  NaN  2015  NaN  NaN  NaN  NaN  
    ====================================================================================================
    Contaminant:  SO2
            FECHA  HORA  ACO   AJM   ATI  BJU   CAM  CCA  CHO   CUA  ...   TLI  \
    0  2015-01-01     1  4.0  20.0  16.0  NaN  67.0  4.0  3.0  16.0  ...  31.0   
    1  2015-01-01     2  5.0   9.0  12.0  NaN  41.0  5.0  3.0  15.0  ...  38.0   
    2  2015-01-01     3  9.0   6.0   9.0  NaN  30.0  4.0  2.0  15.0  ...  71.0   
    
       TPN   UAX   UIZ    VIF   XAL  Year  MPA  FAR  SAC  
    0  2.0   4.0   5.0   86.0  21.0  2015  NaN  NaN  NaN  
    1  2.0   6.0  12.0   80.0  22.0  2015  NaN  NaN  NaN  
    2  2.0  19.0  33.0  133.0  24.0  2015  NaN  NaN  NaN  
    
    [3 rows x 38 columns]
    ====================================================================================================



```python
# Display the number of columns for each pollutant
for contaminant in contaminants:
    print("Contaminant: ", contaminant)
    print(contaminants_data[contaminant].columns.shape)
    print("="*100)
```

    Contaminant:  CO
    (37,)
    ====================================================================================================
    Contaminant:  NO
    (40,)
    ====================================================================================================
    Contaminant:  NO2
    (41,)
    ====================================================================================================
    Contaminant:  NOX
    (40,)
    ====================================================================================================
    Contaminant:  O3
    (42,)
    ====================================================================================================
    Contaminant:  PM10
    (34,)
    ====================================================================================================
    Contaminant:  PM25
    (30,)
    ====================================================================================================
    Contaminant:  PMCO
    (20,)
    ====================================================================================================
    Contaminant:  SO2
    (38,)
    ====================================================================================================


We can observe that each pollutant has a different number of columns, indicating varying numbers of monitoring stations. Additionally, there are numerous null values in the data. To address this issue, we will calculate the average concentration measured by each monitoring station over the period from 2015 to 2023, and use this to fill in these null data points.


```python
columns_to_exclude = ['FECHA', 'HORA', 'Year']
for contaminant in contaminants:
    col_mean = contaminants_data[contaminant].drop(columns=columns_to_exclude).mean().round(1)
    
    for col in contaminants_data[contaminant].columns:
        if col not in columns_to_exclude:
            contaminants_data[contaminant][col] = contaminants_data[contaminant][col].fillna(col_mean[col])
```

With our complete dataset, we can begin exploring some questions.

## How has the concentration of each pollutant changed over the years (from 2015 to 2023)?

To visually represent the evolution of each pollutant's concentration during this period, we will calculate the annual average concentrations. This will allow us to clearly observe trends and changes over time in a concise manner.


```python
n_rows = np.ceil(len(contaminants) / 3).astype(int)

fig, axs = plt.subplots(n_rows, 3, figsize=(15, n_rows*5))
axs = axs.flatten()  

for i, contaminant in enumerate(contaminants):
    numeric_columns = contaminants_data[contaminant].drop(columns = ['HORA']).select_dtypes(include=[np.number]).columns.tolist()

    mean = contaminants_data[contaminant][numeric_columns].groupby('Year').mean()

    for col in mean.columns:
        axs[i].plot(mean.index, mean[col], alpha=0.5, linewidth=1)

    overall_mean = mean.mean(axis=1)

    axs[i].plot(overall_mean.index, overall_mean, label='Overall Mean', linewidth=4, linestyle='--', alpha=1, color='blue')

    axs[i].set_title(f'{contaminant} per year')
    axs[i].set_xlabel('Year')
    axs[i].set_ylabel(f'{contaminant} mean concentration')
    axs[i].grid(True)
    axs[i].legend()

for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()

```


    
![png](../../images/year.png)
    

As observed in the graphs, most pollutants exhibit relatively stable behaviors over time. However, ozone ($$O_{3}$$) shows a slight upward trend.

Another notable observation is the decline in pollutant levels between 2020 and 2021, coinciding with the Covid-19 pandemic. This suggests that lockdown measures and reduced vehicular and transportation usage in the city and globally contributed to lowering pollutant concentrations.


## How does pollutant concentration vary throughout the day?

Similar to the previous question, we can analyze how pollutant concentrations vary throughout the day in the city.

```python
n_rows = np.ceil(len(contaminants) / 3).astype(int)

fig, axs = plt.subplots(n_rows, 3, figsize=(15, n_rows*5))
axs = axs.flatten()  
for i, contaminant in enumerate(contaminants):
    numeric_columns = contaminants_data[contaminant].drop(columns = ['Year']).select_dtypes(include=[np.number]).columns.tolist()

    mean = contaminants_data[contaminant][numeric_columns].groupby('HORA').mean()

    for col in mean.columns:
        axs[i].plot(mean.index, mean[col], alpha=0.5, linewidth=1)

    overall_mean = mean.mean(axis=1)

    axs[i].plot(overall_mean.index, overall_mean, label='Overall Mean', linewidth=4, linestyle='--', alpha=1, color='blue')

    axs[i].set_title(f'{contaminant} per hour')
    axs[i].set_xlabel('Hour')
    axs[i].set_ylabel(f'{contaminant} mean concentration')
    axs[i].grid(True)
    axs[i].legend()

for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()

```


    
![png](../../images/day.png)
    


We can observe several important aspects:

1. Pollutants generated from fuel combustion, such as nitrogen oxides, carbon monoxide, and sulfur dioxide, exhibit distinct peaks between 5 AM and 10 AM, coinciding with peak hours of city mobility.

2. Particles like $$PM_{2.5}$$, $$PM_{10}$$, and coarse particulate matter ($$PM_{co}$$) also show peaks during high mobility periods, specifically between 5 AM and 10 AM, and between 3 PM and 8 PM.

3. Ozone presents a particularly interesting pattern, with a peak around 3 PM, aligning with the time of day when UV radiation is highest. This is consistent as ozone formation is facilitated by UV radiation.

## Is there any correlation between the levels of different pollutants?

We can also explore correlations between the levels of different pollutants. This analysis will help us better understand their interactions and how variations in one pollutant may influence others.

```python
data = {}

# Para cada contaminante, calcula la media diaria
for contaminant in contaminants:
    data[contaminant] = contaminants_data[contaminant].drop(columns=columns_to_exclude).mean()

#print(data)

df = pd.DataFrame(data)

correlation = df.corr()

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(correlation, annot=True, fmt=".2f", linewidths=.5, ax=ax, cmap='coolwarm')

ax.xaxis.tick_top()
ax.yaxis.tick_left()

plt.show()
```


    
![png](../../images/correlation_matrix.png)
    


Given the variety of nitrogen oxides ($$NO_{x}$$) in our dataset, it's clear there is a correlation among them. Furthermore, we observe a strong positive correlation between these nitrogen oxides and carbon monoxide ($$CO$$). Generally, all pollutants analyzed exhibit a positive correlation, with most exceeding a coefficient of 0.5.

In contrast, ozone ($$O_{3}$$) shows a negative correlation with the other pollutants, suggesting that its formation and concentration are influenced by different factors or processes that oppose those affecting the other pollutants.

