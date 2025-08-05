import pandas as pd

fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

fake_df['label'] = 1
true_df['label'] = 0

df = pd.concat([fake_df, true_df], ignore_index=True)

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('data/combined_news.csv', index=False)
print("Data prepared and saved!")