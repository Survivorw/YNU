import tushare as ts



df_industry = ts.get_industry_classified()
df_industry.to_csv("./data/stock_industry_prep.csv", index=False, sep=',')

df_concept = ts.get_concept_classified()
df_concept.to_csv("./data/stock_concept_prep.csv", index=False, sep=',')

df_exevutive= ts.get_executive_classified()
df_exevutive.to_csv("./data/executive_prep.csv", index=False, sep=',')